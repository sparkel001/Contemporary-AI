"""Grad-CAM for true and predicted classes of the same image."""
import time,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from train import model,preprocess,save_json

def heatmap(score,activation):
    gradient,=torch.autograd.grad(score,activation,retain_graph=True)
    cam=(gradient.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True).relu()
    cam=torch.nn.functional.interpolate(cam,(32,32),mode='bilinear',align_corners=False).detach().numpy().squeeze()
    return cam/max(float(cam.max()),1e-10)

def main():
    start=time.perf_counter()
    ds=CIFAR10('data',train=False)
    net=model('resnet').eval()
    net.load_state_dict(torch.load('checkpoints/resnet_basic_0.1_42_72/best.pt',map_location='cpu',weights_only=True))
    stats=json.loads(Path('config/data.json').read_text())
    norm=[torch.tensor(stats[k]).reshape(1,3,1,1) for k in ('mean','std')]
    captured={}
    handle=net.layer4.register_forward_hook(lambda module,inputs,output:captured.update(activation=output))
    logits=net(preprocess(torch.from_numpy(ds.data[1884:1885]),*norm,'none'))
    predicted=int(logits.argmax(1))
    assert ds.targets[1884]==7 and predicted==5
    true_cam=heatmap(logits[0,7],captured['activation'])
    predicted_cam=heatmap(logits[0,5],captured['activation'])
    handle.remove()
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    axes[0].imshow(ds.data[1884]);axes[0].set_title('Original: true horse / predicted dog')
    axes[1].imshow(true_cam,cmap='Reds',vmin=0,vmax=1);axes[1].set_title('True target: horse')
    axes[2].imshow(predicted_cam,cmap='Blues',vmin=0,vmax=1);axes[2].set_title('Predicted target: dog')
    axes[3].imshow(ds.data[1884])
    axes[3].imshow(true_cam,cmap='Reds',alpha=true_cam*.55,vmin=0,vmax=1)
    axes[3].imshow(predicted_cam,cmap='Blues',alpha=predicted_cam*.55,vmin=0,vmax=1)
    axes[3].set_title('Dual overlay: horse red / dog blue')
    for ax in axes:ax.axis('off')
    fig.tight_layout();fig.savefig('results/figures/gradcam_dual.png',dpi=220);plt.close(fig)
    np.savez('results/gradcam_dual_heatmaps.npz',horse=true_cam,dog=predicted_cam)
    save_json(Path('results/gradcam_dual_metadata.json'),dict(index=1884,true_label=7,true_class='horse',predicted_label=5,
        predicted_class='dog',confidence=float(logits.softmax(1)[0,5].detach()),target_layer='layer4',
        method='Class logit gradients; spatial gradient mean; weighted activation sum; ReLU; bilinear resize; independent max normalization',
        true_heatmap_all_zero=bool(not np.any(true_cam)),predicted_heatmap_all_zero=bool(not np.any(predicted_cam)),
        zero_map_explanation='No positive Grad-CAM response after ReLU; zero maps are retained without fabricated normalization',
        seconds=time.perf_counter()-start))

if __name__=='__main__':
    main()

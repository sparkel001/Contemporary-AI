from project_paths import portable_path, posix_path
"""Render figures and report from the completed experiment suite."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from train import model, preprocess

figures = Path('results/reproduced_figures')
figures.mkdir(exist_ok=True)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def finish(name):
    plt.tight_layout()
    plt.savefig(figures/(name+'.png'),dpi=220,bbox_inches='tight')
    plt.close()


def main():
    chosen = portable_path(read(Path('results/selection.json'))['folder'])
    folders = sorted(p for p in chosen.parent.glob('*') if (p/'test.json').exists()
                     )
    records = {p:read(p/'test.json') for p in folders}
    configs = {p:read(p/'config.json') for p in folders}
    history = {p:read(p/'history.json') for p in folders}
    base = [p for p,c in configs.items() if '_wd' not in p.name and c['epochs']==24 and c['seed']==42 and c['augmentation']=='basic' and c['smoothing']==0]
    for metric in ('train_loss','val_loss','train_accuracy','val_accuracy'):
        for p in base:
            plt.plot([r['epoch'] for r in history[p]],[r[metric] for r in history[p]],label=configs[p]['model'])
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_',' '))
        plt.title(metric.replace('_',' ').title())
        plt.legend()
        finish(metric)
    comparisons = dict(models=(base,'model'), augmentation=([p for p,c in configs.items() if '_wd' not in p.name and c['epochs']==24 and c['model']=='resnet' and c['seed']==42 and c['smoothing']==0],'augmentation'),
                       attention=([p for p in base if configs[p]['model']!='cnn'],'model'))
    for title,(group,key) in comparisons.items():
        x = np.arange(len(group))
        plt.bar(x-.2,[max(r['val_accuracy'] for r in history[p]) for p in group],.4,label='Validation')
        plt.bar(x+.2,[records[p]['accuracy'] for p in group],.4,label='Test')
        plt.xticks(x,[configs[p][key] for p in group])
        plt.ylabel('Accuracy')
        plt.title(title.title()+' comparison')
        plt.legend()
        finish(title)
    ds = CIFAR10('data',train=False)
    classes = ds.classes
    rec = records[chosen]
    matrix = np.array(rec['matrix'])
    plt.figure(figsize=(8,7))
    plt.imshow(matrix,cmap='Blues')
    for (i,j),value in np.ndenumerate(matrix):
        plt.text(j,i,str(value),ha='center',va='center',color='white' if value>matrix.max()/2 else 'black',fontsize=8)
    plt.xticks(range(10),classes,rotation=45)
    plt.yticks(range(10),classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Test confusion matrix')
    finish('confusion')
    for name,metrics in [('classification',['precision','recall','f1-score']),('per_class',['recall'])]:
        for i,key in enumerate(metrics):
            plt.bar(np.arange(10)+i*.24,[rec['metrics'][c][key] for c in classes],.24,label=key)
        plt.xticks(range(10),classes,rotation=50)
        plt.ylabel('Score')
        plt.title('Class metrics' if len(metrics)>1 else 'Per-class accuracy')
        plt.legend()
        finish(name)
    preds = np.load(chosen/'predictions.npz')
    y,pred,conf = [preds[k] for k in ('target','pred','confidence')]
    wrong = np.where(y!=pred)[0]
    examples = wrong[np.argsort(conf[wrong])[-6:][::-1]]
    plt.figure(figsize=(9,6))
    for i,idx in enumerate(examples):
        plt.subplot(2,3,i+1)
        plt.imshow(ds.data[idx])
        plt.title(f'{classes[y[idx]]} → {classes[pred[idx]]}\np={conf[idx]:.3f}')
        plt.axis('off')
    finish('errors')
    net = model(configs[chosen]['model'])
    net.load_state_dict(torch.load(chosen/'best.pt',map_location='cpu'))
    net.eval()
    layer = net.layer4 if hasattr(net,'layer4') else net[-4]
    items = {}
    def capture(module, inputs, output):
        items['activation'] = output
    handle = layer.register_forward_hook(capture)
    norm = read(Path('config/data.json'))
    stats = [torch.tensor(norm[k]).reshape(1,3,1,1) for k in ('mean','std')]
    plt.figure(figsize=(8,5))
    cam_notes = []
    for row,idx in enumerate([np.flatnonzero(y==pred)[0],examples[0]]):
        x = preprocess(torch.from_numpy(ds.data[idx:idx+1]),*stats,'none')
        logits = net(x)
        act = items['activation']
        grad, = torch.autograd.grad(logits[0,pred[idx]],act)
        cam = (grad.mean((2,3),keepdim=True)*act).sum(1,keepdim=True).relu()
        heat = torch.nn.functional.interpolate(cam,(32,32),mode='bilinear',align_corners=False).detach().numpy().squeeze()
        heat /= max(float(heat.max()),1e-10)
        point = np.unravel_index(heat.argmax(),heat.shape)
        cam_notes.append(f'样本 {idx} 的真实类别为 {classes[y[idx]]}，预测为 {classes[pred[idx]]}，响应峰值位于第 {point[0]+1} 行、第 {point[1]+1} 列。')
        for col,title in enumerate(('Original','Grad-CAM','Overlay')):
            plt.subplot(2,3,row*3+col+1)
            if col!=1:
                plt.imshow(ds.data[idx])
            if col!=0:
                plt.imshow(heat,cmap='jet',alpha=.45 if col==2 else 1,vmin=0,vmax=1)
            plt.axis('off')
            plt.title(title+' #'+str(idx))
    handle.remove()
    finish('gradcam')


if __name__ == '__main__':
    main()

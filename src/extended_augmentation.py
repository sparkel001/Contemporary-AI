"""Paired exploratory augmentation experiment; never updates selection."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from train import fit,tensor_data,run_epoch,model,save_json

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_json(Path('results/augmentation_72_protocol.json'),dict(model='resnet',augmentation='cutout',smoothing=.1,epochs=72,seeds=[42,43,44],batch=256,weight_decay=.01,optimizer='AdamW',learning_rate=.003,scheduler='CosineAnnealingLR',eta_min=.00005,split_seed=42,device=str(device),torch=torch.__version__))
    ds=CIFAR10('data')
    a,b=train_test_split(np.arange(len(ds)),test_size=5000,stratify=ds.targets,random_state=42)
    train,val=tensor_data(ds,a,device),tensor_data(ds,b,device)
    stats=json.loads(Path('config/data.json').read_text())
    norm=[torch.tensor(stats[k],device=device).reshape(1,3,1,1) for k in ('mean','std')]
    folders=[fit(('resnet','cutout',.1,seed,72),train,val,norm,256)[0] for seed in (42,43,44)]
    test=tensor_data(CIFAR10('data',train=False),slice(None),device)
    for folder in folders:
        if (folder/'test.json').exists():
            continue
        net=model('resnet').to(device,memory_format=torch.channels_last)
        net.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True))
        start=time.perf_counter()
        scores,pred,conf=run_epoch(net,test,norm,256)
        pred=pred.cpu().numpy()
        elapsed=time.perf_counter()-start
        y=test[1].cpu().numpy()
        np.savez(folder/'predictions.npz',target=y,pred=pred,confidence=conf.cpu().numpy())
        save_json(folder/'test.json',dict(accuracy=float((y==pred).mean()),loss=scores[0],evaluation_seconds=elapsed,
            parameters=sum(p.numel() for p in net.parameters()),matrix=confusion_matrix(y,pred).tolist(),
            metrics=classification_report(y,pred,target_names=ds.classes,output_dict=True,zero_division=0)))
    result={'interpretation':'Exploratory test evaluation; fixed protocol, no test-based selection','pairs':[]}
    for seed in (42,43,44):
        pair={'seed':seed}
        for aug in ('basic','cutout'):
            folder=Path(f'checkpoints/resnet_{aug}_0.1_{seed}_72')
            history=json.loads((folder/'history.json').read_text())
            best=max(history,key=lambda x:x['val_accuracy'])
            pair[aug]=dict(validation_accuracy=best['val_accuracy'],test_accuracy=json.loads((folder/'test.json').read_text())['accuracy'],
                best_epoch=best['epoch'],training_seconds=sum(x['seconds'] for x in history))
        pair['test_difference']=pair['cutout']['test_accuracy']-pair['basic']['test_accuracy']
        pair['validation_difference']=pair['cutout']['validation_accuracy']-pair['basic']['validation_accuracy']
        result['pairs'].append(pair)
    for aug in ('basic','cutout'):
        result[aug]={k:dict(mean=float(np.mean([p[aug][k] for p in result['pairs']])),sample_std=float(np.std([p[aug][k] for p in result['pairs']],ddof=1))) for k in ('validation_accuracy','test_accuracy')}
    save_json(Path('results/augmentation_72_comparison.json'),result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key in zip(axes,('validation_accuracy','test_accuracy')):
        for aug in ('basic','cutout'):
            ax.plot([42,43,44],[p[aug][key] for p in result['pairs']],marker='o',label=aug)
        ax.set(xlabel='Initialization seed',ylabel='Accuracy',title=key.replace('_',' ').title(),xticks=[42,43,44])
        ax.legend()
    fig.tight_layout()
    fig.savefig('results/figures/augmentation_72.png',dpi=220)
    plt.close(fig)

if __name__=='__main__':
    main()

from project_paths import portable_path, posix_path
"""Predeclared longer schedule for the validation-selected configuration."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.datasets import CIFAR10
from train import fit, tensor_data, run_epoch, model, save_json, update_selection
import time

def main():
    choice = json.loads(Path('results/selection.json').read_text())
    short = Path('checkpoints') / ('_'.join(map(str,choice['setting']))+'_42_24')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = CIFAR10('data')
    a,b = train_test_split(np.arange(len(ds)),test_size=5000,stratify=ds.targets,random_state=42)
    train,val = tensor_data(ds,a,device),tensor_data(ds,b,device)
    stats = json.loads(Path('config/data.json').read_text())
    norm = [torch.tensor(stats[k],device=device).reshape(1,3,1,1) for k in ('mean','std')]
    batch = json.loads((short/'config.json').read_text())['batch']
    runs = [fit((*choice['setting'],seed,72),train,val,norm,batch) for seed in (42,43,44)]
    if runs[0][1] >= max(r['val_accuracy'] for r in json.loads((short/'history.json').read_text())):
        choice['folder'] = posix_path(runs[0][0])
    else:
        choice['folder'] = posix_path(short)
    update_selection(choice)
    data = tensor_data(CIFAR10('data',train=False),slice(None),device)
    for folder,_ in runs:
        net = model(choice['setting'][0]).to(device,memory_format=torch.channels_last)
        net.load_state_dict(torch.load(folder/'best.pt',map_location=device))
        run_epoch(net,(data[0][:batch],data[1][:batch]),norm,batch)
        start=time.perf_counter()
        scores,pred,conf=run_epoch(net,data,norm,batch)
        pred=pred.cpu().numpy()
        sec=time.perf_counter()-start
        y=data[1].cpu().numpy()
        np.savez(folder/'predictions.npz',target=y,pred=pred,confidence=conf.cpu().numpy())
        save_json(folder/'test.json',dict(accuracy=float((pred==y).mean()),loss=scores[0],
                  evaluation_seconds=sec,parameters=sum(p.numel() for p in net.parameters()),
                  matrix=confusion_matrix(y,pred).tolist(),metrics=classification_report(y,pred,target_names=ds.classes,output_dict=True,zero_division=0)))

if __name__=='__main__':
    main()

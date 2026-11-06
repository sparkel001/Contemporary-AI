from project_paths import portable_path, posix_path
"""Execute from the project directory: python src/train.py."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split

from models import model


def save_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf8')
    temp.replace(path)


def update_selection(candidate):
    candidate = dict(candidate, folder=posix_path(candidate['folder']))
    path = Path('results/selection.json')
    def score(choice):
        history = json.loads((portable_path(choice['folder']) / 'history.json').read_text())
        return max(row['val_accuracy'] for row in history)
    if not path.exists() or score(candidate) > score(json.loads(path.read_text())):
        save_json(path, candidate)


def tensor_data(dataset, ids, device):
    images = torch.from_numpy(dataset.data[ids]).to(device)
    labels = torch.tensor(np.asarray(dataset.targets)[ids], device=device, dtype=torch.long)
    return images, labels


def preprocess(x, mean, std, aug):
    x = x.permute(0, 3, 1, 2).float().div_(255)
    if aug != 'none':
        count = len(x)
        padded = F.pad(x, (4, 4, 4, 4)).permute(0, 2, 3, 1)
        offset = torch.randint(9, (2, count, 1, 1), device=x.device)
        axis = torch.arange(32, device=x.device)
        cols = torch.where(torch.rand(count, 1, 1, device=x.device) < .5, axis.flip(0), axis)
        x = padded[torch.arange(count, device=x.device)[:, None, None],
                   offset[0] + axis[:, None], offset[1] + cols].permute(0, 3, 1, 2)
    x = (x - mean) / std
    if aug == 'cutout':
        origin = torch.randint(25, (2, len(x), 1, 1), device=x.device)
        dy = axis[:, None] - origin[0]
        dx = axis - origin[1]
        x = x.masked_fill(((dy >= 0) & (dy < 8) & (dx >= 0) & (dx < 8))[:, None], 0)
    return x.contiguous(memory_format=torch.channels_last)


def run_epoch(net, data, norm, batch, aug='none', opt=None, smoothing=0):
    images, labels = data
    device = images.device
    net.train(opt is not None)
    order = torch.randperm(len(labels), device=device) if opt else torch.arange(len(labels), device=device)
    totals = torch.zeros(2, device=device)
    guesses, confidence = [], []
    with torch.set_grad_enabled(opt is not None):
        for ids in order.split(batch):
            x = preprocess(images[ids], *norm, aug)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda' and torch.cuda.is_bf16_supported()):
                logits = net(x)
            loss = F.cross_entropy(logits.float(), labels[ids], label_smoothing=smoothing)
            if opt:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            prob, pred = logits.detach().float().softmax(1).max(1)
            totals += torch.stack((loss.detach() * len(ids), (pred == labels[ids]).sum()))
            guesses.append(pred)
            confidence.append(prob)
    return (totals / len(labels)).tolist(), torch.cat(guesses), torch.cat(confidence)


def fit(cfg, train, val, norm, batch, weight_decay=.01):
    name, aug, smoothing, seed, epochs = cfg
    folder = Path('checkpoints') / f'{name}_{aug}_{smoothing}_{seed}_{epochs}'
    if weight_decay != .01:
        folder = folder.with_name(folder.name + f'_wd{weight_decay}')
    folder.mkdir(parents=True, exist_ok=True)
    requested = dict(zip(('model','augmentation','smoothing','seed','epochs'),cfg), batch=batch, weight_decay=weight_decay)
    config_path = folder / 'config.json'
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing != requested:
            raise ValueError(f'Existing experiment configuration differs: {folder}. Use a new experiment directory.')
    elif any(folder.iterdir()):
        raise ValueError(f'Existing experiment evidence has no configuration: {folder}')
    else:
        save_json(config_path, requested)
    if (folder / 'history.json').exists():
        completed = json.loads((folder / 'history.json').read_text())
        if len(completed) == epochs and (folder / 'best.pt').exists():
            return folder, max(row['val_accuracy'] for row in completed)
    device = train[0].device
    torch.manual_seed(seed)
    net = model(name).to(device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=.003, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=.00005)
    history = []
    state_file = folder / 'resume.pt'
    if state_file.exists():
        state = torch.load(state_file, map_location=device)
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        history = state['history']
    for epoch in range(len(history), epochs):
        torch.manual_seed(seed * 1000 + epoch)
        start = time.perf_counter()
        lr = opt.param_groups[0]['lr']
        tr, _, _ = run_epoch(net, train, norm, batch, aug, opt, smoothing)
        va, _, _ = run_epoch(net, val, norm, batch)
        scheduler.step()
        if not history or va[1] > max(r['val_accuracy'] for r in history):
            torch.save(net.state_dict(), folder / 'best.tmp')
            (folder / 'best.tmp').replace(folder / 'best.pt')
        history.append(dict(epoch=epoch+1, train_loss=tr[0], train_accuracy=tr[1], val_loss=va[0],
                            val_accuracy=va[1], lr=lr, seconds=time.perf_counter()-start))
        torch.save(dict(model=net.state_dict(), optimizer=opt.state_dict(), scheduler=scheduler.state_dict(),
                        history=history), folder / 'resume.tmp')
        (folder / 'resume.tmp').replace(state_file)
        print(folder.name, history[-1], flush=True)
    save_json(folder / 'history.json', history)
    return folder, max(r['val_accuracy'] for r in history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=24)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = 256 if device.type == 'cuda' and torch.cuda.mem_get_info()[0] > 4*1024**3 else 64
    ds = CIFAR10('data', download=True)
    a, b = train_test_split(np.arange(len(ds)), test_size=5000, stratify=ds.targets, random_state=42)
    train, val = tensor_data(ds, a, device), tensor_data(ds, b, device)
    # Accumulate moments over training pixels only, in bounded chunks.
    chunks = [np.stack((v.mean((0,1,2)), (v*v).mean((0,1,2)))) for ids in np.array_split(a,90)
              for v in [ds.data[ids].astype(np.float64)/255]]
    mean, second = np.mean(chunks,axis=0)
    std = np.sqrt(second-mean**2)
    save_json(Path('config/data.json'), dict(mean=mean.tolist(),std=std.tolist(),split_seed=42,classes=ds.classes))
    norm = [torch.tensor(v,device=device,dtype=torch.float32).reshape(1,3,1,1) for v in (mean,std)]
    experiments = [(m,'basic',0) for m in ('cnn','resnet','se')]
    experiments += [('resnet',aug,0) for aug in ('none','cutout')] + [('resnet','basic',.1)]
    runs = {}
    for setting in experiments:
        runs[setting] = fit((*setting,42,args.epochs),train,val,norm,batch)
    winner = max(runs,key=lambda s:runs[s][1])
    update_selection(dict(setting=winner,folder=posix_path(runs[winner][0])))
    folders = [v[0] for v in runs.values()]
    for seed in (43,44):
        folders.append(fit((*winner,seed,args.epochs),train,val,norm,batch)[0])
    test = tensor_data(CIFAR10('data',train=False),slice(None),device)
    from sklearn.metrics import confusion_matrix, classification_report
    for folder in folders:
        cfg = json.loads((folder/'config.json').read_text())
        net = model(cfg['model']).to(device,memory_format=torch.channels_last)
        net.load_state_dict(torch.load(folder/'best.pt',map_location=device))
        run_epoch(net,(test[0][:batch],test[1][:batch]),norm,batch)
        start = time.perf_counter()
        scores,pred,prob = run_epoch(net,test,norm,batch)
        pred = pred.cpu().numpy()
        seconds = time.perf_counter()-start
        target = test[1].cpu().numpy()
        np.savez(folder/'predictions.npz',target=target,pred=pred,confidence=prob.cpu().numpy())
        save_json(folder/'test.json',dict(accuracy=float((pred==target).mean()), loss=scores[0],
                  evaluation_seconds=seconds, parameters=sum(p.numel() for p in net.parameters()),
                  matrix=confusion_matrix(target,pred).tolist(),
                  metrics=classification_report(target,pred,target_names=ds.classes,output_dict=True,zero_division=0)))


if __name__ == '__main__':
    main()


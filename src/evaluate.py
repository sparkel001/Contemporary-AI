from project_paths import portable_path, posix_path
"""Re-evaluate an individual saved checkpoint on official test data."""
import argparse
import json
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10
from train import model, run_epoch, tensor_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', nargs='?', type=portable_path)
    parser.add_argument('--data-root', default='data')
    args = parser.parse_args()
    folder = args.folder or portable_path(json.loads(Path('results/selection.json').read_text())['folder'])
    cfg = json.loads((folder/'config.json').read_text())
    stats = json.loads(Path('config/data.json').read_text())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model(cfg['model']).to(device,memory_format=torch.channels_last)
    net.load_state_dict(torch.load(folder/'best.pt',map_location=device))
    norm = [torch.tensor(stats[k],device=device).reshape(1,3,1,1) for k in ('mean','std')]
    data = tensor_data(CIFAR10(args.data_root,train=False,download=True),slice(None),device)
    (loss,acc),_,_ = run_epoch(net,data,norm,cfg['batch'])
    print(f'loss={loss:.6f}  accuracy={acc:.6f}  n={len(data[1])}')


if __name__=='__main__':
    main()

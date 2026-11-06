"""Time warmed-up model forward passes independently of preprocessing."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from train import model,save_json

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=torch.zeros(128,3,32,32,device=device).contiguous(memory_format=torch.channels_last)
    output={}
    with torch.inference_mode(),torch.autocast(device.type,dtype=torch.bfloat16,enabled=device.type=='cuda' and torch.cuda.is_bf16_supported()):
        for name in ('cnn','resnet','se'):
            folder=Path('checkpoints')/f'{name}_basic_0_42_24'
            net=model(name).to(device,memory_format=torch.channels_last).eval()
            net.load_state_dict(torch.load(folder/'best.pt',map_location=device))
            for _ in range(20):
                net(x)
            timings=[]
            for _ in range(10):
                if device.type=='cuda':
                    torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(20):
                    net(x)
                if device.type=='cuda':
                    torch.cuda.synchronize()
                timings.append((time.perf_counter()-start)*1000/20)
            output[name]=dict(batch=128,median_ms=float(np.median(timings)),samples_ms=timings)
    save_json(Path('results/inference.json'),output)

if __name__=='__main__':
    main()

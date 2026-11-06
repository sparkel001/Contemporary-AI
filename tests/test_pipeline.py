import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from train import model, run_epoch


def test_train_eval_and_reload(tmp_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.randint(256,(8,32,32,3),dtype=torch.uint8,device=device)
    labels = torch.arange(8,device=device)
    for name, aug in [('cnn','none'),('resnet','basic'),('se','cutout')]:
        net = model(name).to(device)
        run_epoch(net,(images,labels),(0,1),4,aug,torch.optim.AdamW(net.parameters()))
        scores,pred,prob = run_epoch(net,(images,labels),(0,1),4)
        assert scores[1] == (pred==labels).float().mean().item()
        assert prob.min() >= .1
        torch.save(net.state_dict(),tmp_path/'weights.pt')
        restored = model(name).to(device)
        restored.load_state_dict(torch.load(tmp_path/'weights.pt',map_location=device))
        assert torch.equal(run_epoch(restored,(images,labels),(0,1),4)[1],pred)

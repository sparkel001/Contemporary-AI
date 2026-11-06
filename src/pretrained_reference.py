"""Fixed ImageNet transfer reference, selected only on validation accuracy."""
import json,time,platform,os
from pathlib import Path
import numpy as np
import torch,torchvision
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
from torchvision.datasets import CIFAR10
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from train import tensor_data,preprocess,save_json

def reference_model():
    net=resnet18(weights=None)
    net.fc=torch.nn.Linear(net.fc.in_features,10)
    return net

def epoch(net,data,batch,norm,opt=None):
    net.train(opt is not None)
    x,y=data
    order=torch.randperm(len(y),device=y.device) if opt else torch.arange(len(y),device=y.device)
    total=0.;correct=0;predictions=[];confidence=[]
    with torch.set_grad_enabled(opt is not None):
        for ids in order.split(batch):
            inputs=preprocess(x[ids],0,1,'basic' if opt else 'none')
            inputs=F.interpolate(inputs,(224,224),mode='bilinear',align_corners=False,antialias=True)
            inputs=(inputs-norm[0])/norm[1]
            with torch.autocast(y.device.type,dtype=torch.bfloat16,enabled=y.device.type=='cuda' and torch.cuda.is_bf16_supported()):
                logits=net(inputs)
            loss=F.cross_entropy(logits.float(),y[ids])
            if opt:
                opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            prob,pred=logits.detach().float().softmax(1).max(1)
            total+=float(loss.detach())*len(ids);correct+=int((pred==y[ids]).sum())
            predictions.append(pred.cpu());confidence.append(prob.cpu())
    return dict(loss=total/len(y),accuracy=correct/len(y)),torch.cat(predictions).numpy(),torch.cat(confidence).numpy()

def main():
    folder=Path('checkpoints/pretrained_resnet18_imagenet_42')
    folder.mkdir(exist_ok=True)
    torch.hub.set_dir(str(Path('data/torch_hub').resolve()))
    torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=32 if device.type=='cuda' and torch.cuda.mem_get_info()[0]>4*1024**3 else 8
    start=time.perf_counter()
    try:
        net=resnet18(weights=ResNet18_Weights.DEFAULT)
    except Exception as error:
        save_json(Path('results/pretrained_reference.json'),dict(status='failed',reason=repr(error),seconds=time.perf_counter()-start))
        raise
    net.fc=torch.nn.Linear(net.fc.in_features,10)
    net=net.to(device,memory_format=torch.channels_last)
    transform=ResNet18_Weights.DEFAULT.transforms()
    norm=[torch.tensor(v,device=device).reshape(1,3,1,1) for v in (transform.mean,transform.std)]
    config=dict(model='pretrained_resnet18',seed=42,epochs=12,patience=3,batch=batch,weights=str(ResNet18_Weights.DEFAULT),
        weight_url=ResNet18_Weights.DEFAULT.url,input_size=224,mean=transform.mean,std=transform.std,
        augmentation='RandomCrop32 padding4; horizontal flip; bilinear antialiased resize224',
        validation_transform='deterministic bilinear antialiased resize224 and ImageNet normalization',
        optimizer='AdamW',backbone_lr=.0001,classifier_lr=.001,weight_decay=.01,scheduler='CosineAnnealingLR',eta_min=.000001,
        python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory if device.type=='cuda' else None,
        cpu_threads=os.cpu_count(),split_seed=42)
    if (folder/'config.json').exists():
        assert json.loads((folder/'config.json').read_text())==config
    else:save_json(folder/'config.json',config)
    ds=CIFAR10('data')
    a,b=train_test_split(np.arange(len(ds)),test_size=5000,stratify=ds.targets,random_state=42)
    train,val=tensor_data(ds,a,device),tensor_data(ds,b,device)
    opt=torch.optim.AdamW([dict(params=[p for n,p in net.named_parameters() if not n.startswith('fc.')],lr=.0001),
                          dict(params=net.fc.parameters(),lr=.001)],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,12,eta_min=.000001)
    history=[];best=-1;stale=0
    if (folder/'resume.pt').exists():
        saved=torch.load(folder/'resume.pt',map_location=device,weights_only=False)
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        history,best,stale=saved['history'],saved['best'],saved['stale']
    for i in range(len(history),12):
        if stale>=3:break
        torch.manual_seed(42000+i)
        tick=time.perf_counter()
        learning_rates=[group['lr'] for group in opt.param_groups]
        tr,_,_=epoch(net,train,batch,norm,opt);va,_,_=epoch(net,val,batch,norm)
        if va['accuracy']>best:
            best=va['accuracy'];stale=0
            torch.save(net.state_dict(),folder/'best.tmp');(folder/'best.tmp').replace(folder/'best.pt')
        else:stale+=1
        scheduler.step()
        history.append(dict(epoch=i+1,train_loss=tr['loss'],train_accuracy=tr['accuracy'],val_loss=va['loss'],val_accuracy=va['accuracy'],
                            learning_rates=learning_rates,seconds=time.perf_counter()-tick))
        save_json(folder/'history.json',history)
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,best=best,stale=stale),folder/'resume.tmp')
        (folder/'resume.tmp').replace(folder/'resume.pt')
        print(history[-1],flush=True)
    net.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True))
    test=tensor_data(CIFAR10('data',train=False),slice(None),device)
    tick=time.perf_counter();metrics,pred,conf=epoch(net,test,batch,norm);elapsed=time.perf_counter()-tick
    y=test[1].cpu().numpy()
    np.savez(folder/'predictions.npz',target=y,pred=pred,confidence=conf)
    save_json(folder/'test.json',dict(accuracy=metrics['accuracy'],loss=metrics['loss'],parameters=sum(p.numel() for p in net.parameters()),
        evaluation_seconds=elapsed,matrix=confusion_matrix(y,pred).tolist(),
        metrics=classification_report(y,pred,target_names=ds.classes,output_dict=True,zero_division=0)))
    net.eval()
    sample=torch.zeros(batch,3,224,224,device=device)
    with torch.inference_mode():
        for _ in range(5):net(sample)
        if device.type=='cuda':torch.cuda.synchronize()
        tick=time.perf_counter()
        for _ in range(20):net(sample)
        if device.type=='cuda':torch.cuda.synchronize()
    latency=(time.perf_counter()-tick)/20
    save_json(Path('results/pretrained_reference.json'),dict(status='complete',best_validation_accuracy=best,
        best_epoch=max(history,key=lambda r:r['val_accuracy'])['epoch'],test_accuracy=metrics['accuracy'],
        parameters=sum(p.numel() for p in net.parameters()),training_seconds=sum(r['seconds'] for r in history),
        invocation_seconds=time.perf_counter()-start,inference_batch=batch,inference_batch_seconds=latency,
        inference_precision='float32',interpretation='Exploratory transfer reference; different resolution, capacity and budget, not a controlled architecture comparison'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline=json.loads(Path('checkpoints/resnet_basic_0.1_42_72/test.json').read_text())['accuracy']
    plt.bar(['Scratch narrow\n32px / 72 epochs','ImageNet ResNet18\n224px / <=12 epochs'],[baseline,metrics['accuracy']])
    plt.ylabel('Exploratory test accuracy');plt.title('Transfer reference: different budgets and inputs')
    plt.tight_layout();plt.savefig('results/figures/pretrained_comparison.png',dpi=220);plt.close()

if __name__=='__main__':
    main()

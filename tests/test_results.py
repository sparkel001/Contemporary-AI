from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import json
import numpy as np
import torch
from models import model
from sklearn.metrics import confusion_matrix, classification_report

def test_packaged_evidence():
    folders=[p for p in Path('checkpoints').glob('*/config.json') if (p.parent/'test.json').exists()]
    assert len(folders)>=15
    classes=json.loads(Path('config/data.json').read_text())['classes']
    for config in folders:
        folder=config.parent
        cfg=json.loads(config.read_text())
        history=json.loads((folder/'history.json').read_text())
        assert 1 <= len(history) <= cfg['epochs']
        if cfg['model'] != 'pretrained_resnet18':
            assert len(history)==cfg['epochs']
        assert [r['epoch'] for r in history]==list(range(1,len(history)+1))
        if cfg['model']=='pretrained_resnet18':
            from pretrained_reference import reference_model
            net=reference_model()
        else:
            net=model(cfg['model'])
        net.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=True),strict=True)
        result=json.loads((folder/'test.json').read_text())
        assert sum(p.numel() for p in net.parameters())==result['parameters']
        with np.load(folder/'predictions.npz') as predictions:
            y,pred=predictions['target'],predictions['pred']
            assert len(y)==10000 and ((y>=0)&(y<10)).all()
            assert ((pred>=0)&(pred<10)).all()
            assert np.isfinite(predictions['confidence']).all()
            assert float((y==pred).mean())==result['accuracy']
            assert np.array_equal(confusion_matrix(y,pred),result['matrix'])
            metrics=classification_report(y,pred,target_names=classes,output_dict=True,zero_division=0)
            for c in classes:
                for k in ('precision','recall','f1-score','support'):
                    assert np.isclose(metrics[c][k],result['metrics'][c][k])

def test_selection_does_not_regress(tmp_path,monkeypatch):
    from train import update_selection
    monkeypatch.chdir(tmp_path)
    Path('results').mkdir()
    for folder,score in [('long',.9),('short',.8)]:
        Path(folder).mkdir()
        (Path(folder)/'history.json').write_text(json.dumps([{'val_accuracy':score}]))
    update_selection({'folder':'long'})
    update_selection({'folder':'short'})
    assert json.loads(Path('results/selection.json').read_text())['folder']=='long'

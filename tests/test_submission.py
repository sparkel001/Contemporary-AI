import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import subprocess
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from project_paths import portable_path,posix_path
from train import fit

def test_portable_selection():
    choice=json.loads(Path('results/selection.json').read_text())['folder']
    assert choice=='checkpoints/resnet_basic_0.1_42_72'
    for text in (choice,choice.replace('/',chr(92))):
        value=posix_path(text)
        assert PurePosixPath(value).parts==PureWindowsPath(value).parts
        assert portable_path(text)==Path('checkpoints')/'resnet_basic_0.1_42_72'
        assert (portable_path(text)/'best.pt').is_file()

def test_configuration_preserved(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder=Path('checkpoints/resnet_basic_0_42_24')
    folder.mkdir(parents=True)
    config=dict(model='resnet',augmentation='basic',smoothing=0,seed=42,epochs=24,batch=256,weight_decay=.01)
    (folder/'config.json').write_text(json.dumps(config))
    (folder/'history.json').write_text(json.dumps([{'val_accuracy':.8}]*24))
    (folder/'best.pt').write_bytes(b'existing evidence')
    original={p.name:p.read_bytes() for p in folder.iterdir()}
    with pytest.raises(ValueError,match='configuration differs'):
        fit(('resnet','basic',0,42,24),None,None,None,64)
    assert fit(('resnet','basic',0,42,24),None,None,None,256)==(folder,.8)
    assert original=={p.name:p.read_bytes() for p in folder.iterdir()}

def test_complete_experiment_files():
    folders=list(Path('checkpoints').iterdir())
    assert len(folders)==16
    expected={'config.json','history.json','best.pt','predictions.npz','test.json'}
    for folder in folders:
        assert expected <= {p.name for p in folder.iterdir()}
        cfg=json.loads((folder/'config.json').read_text())
        assert cfg['seed'] in (42,43,44)

def test_report_images():
    import zipfile
    from xml.etree import ElementTree as ET
    with zipfile.ZipFile('report.docx') as doc:
        assert doc.testzip() is None
        ET.fromstring(doc.read('word/document.xml'))
        media={hashlib.sha256(doc.read(n)).digest() for n in doc.namelist() if n.startswith('word/media/')}
    images=list(Path('results/figures').glob('*.png'))
    assert len(images)==15
    assert all(p.read_bytes().startswith(bytes.fromhex('89504e470d0a1a0a')) for p in images)
    assert media & {hashlib.sha256(p.read_bytes()).digest() for p in images}

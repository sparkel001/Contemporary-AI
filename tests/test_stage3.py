from pathlib import Path
import hashlib,json
import numpy as np

def test_selected_model_predictions():
    folder=Path(json.loads(Path('results/selection.json').read_text())['folder'])
    with np.load(folder/'predictions.npz') as data:
        assert (data['target']==data['pred']).sum()==8931
    for pair in json.loads(Path('results/augmentation_72_comparison.json').read_text())['pairs']:
        for aug in ('basic','cutout'):
            folder=Path(f"checkpoints/resnet_{aug}_0.1_{pair['seed']}_72")
            assert pair[aug]['test_accuracy']==json.loads((folder/'test.json').read_text())['accuracy']

def test_paired_augmentation():
    record=json.loads(Path('results/augmentation_72_comparison.json').read_text())
    assert [p['seed'] for p in record['pairs']]==[42,43,44]
    for aug in ('basic','cutout'):
        for metric in ('validation_accuracy','test_accuracy'):
            scores=[p[aug][metric] for p in record['pairs']]
            assert np.isclose(np.mean(scores),record[aug][metric]['mean'])
            assert np.isclose(np.std(scores,ddof=1),record[aug][metric]['sample_std'])
    for p in record['pairs']:
        assert np.isclose(p['test_difference'],p['cutout']['test_accuracy']-p['basic']['test_accuracy'])

def test_dual_cam():
    m=json.loads(Path('results/gradcam_dual_metadata.json').read_text())
    assert (m['index'],m['true_label'],m['predicted_label'])==(1884,7,5)
    maps=np.load('results/gradcam_dual_heatmaps.npz')
    for v in maps.values():
        assert v.shape==(32,32) and np.isfinite(v).all()
        assert v.min()>=0 and v.max()<=1
    assert not np.array_equal(maps['horse'],maps['dog'])

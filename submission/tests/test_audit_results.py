"""Portable recorded paths must retain strict source-content validation."""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_results import source_hash_matches


@pytest.mark.parametrize('recorded_path', [
    r'src\search_lab\sample.py', 'src/search_lab/sample.py',
])
@pytest.mark.parametrize('changed', [False, True])
def test_source_hash_checks_both_path_styles(tmp_path, recorded_path, changed):
    source = tmp_path / 'src' / 'search_lab' / 'sample.py'
    source.parent.mkdir(parents=True)
    original = b'value = 7\n'
    digest = hashlib.sha256(original).hexdigest()
    source.write_bytes(b'value = 8\n' if changed else original)
    assert source_hash_matches(tmp_path, {recorded_path: digest}) == {
        recorded_path: not changed,
    }


@pytest.mark.parametrize('recorded_path', [r'src\missing.py', 'src/missing.py'])
def test_missing_source_is_not_skipped(tmp_path, recorded_path):
    with pytest.raises(FileNotFoundError):
        source_hash_matches(tmp_path, {recorded_path: '0' * 64})

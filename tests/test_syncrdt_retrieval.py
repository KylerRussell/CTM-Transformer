"""Checks for the Sync-RDT retrieval study: data suite and decision logic."""
import hashlib
import json

import pytest

from ctm_transformer.algorithmic import canonical_json
from ctm_transformer.mqar_suite import create_mqar_suite, validate_mqar_suite
from scripts.summarize_syncrdt_retrieval import compare, steps_to


def test_mqar_suite_is_deterministic_disjoint_and_all_keys(tmp_path):
    a = create_mqar_suite(tmp_path / 'a', [3, 5], 64, 8, 8, 'x')
    create_mqar_suite(tmp_path / 'b', [3, 5], 64, 8, 8, 'x')
    for seed, splits in a['seeds'].items():
        for split, entry in splits.items():
            assert (tmp_path / 'a' / entry['path']).read_bytes() == (tmp_path / 'b' / entry['path']).read_bytes()
    data = validate_mqar_suite(tmp_path / 'a')
    record = data['3']['validation'].records[0]
    assert len(record['queries']) == 12 and sorted(record['queries']) == sorted(record['successors'])
    assert data['3']['train'].ids != data['5']['train'].ids
    with pytest.raises(ValueError, match='empty'):
        create_mqar_suite(tmp_path / 'a', [3], 8, 8, 8, 'x')


def test_mqar_suite_rejects_wrong_answers_even_with_refreshed_hash(tmp_path):
    create_mqar_suite(tmp_path / 'a', [3], 16, 8, 8, 'x')
    root = tmp_path / 'a'
    manifest = json.loads((root / 'manifest.json').read_text())
    path = root / manifest['seeds']['3']['evaluation']['path']
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    r = rows[0]
    wrong = 'a' if r['answers'][0] != 'a' else 'b'
    i = r['supervised'][0]
    r['text'] = r['text'][:i] + wrong + r['text'][i + 1:]
    r['answers'][0] = wrong
    path.write_text(''.join(canonical_json(x) + '\n' for x in rows))
    manifest['seeds']['3']['evaluation']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='wrong answer'):
        validate_mqar_suite(root)


def test_threshold_and_censored_paired_decisions():
    curve = [{'step': 250 * i, 'position1': x} for i, x in enumerate([0.1, 0.4, 0.55, 0.92, 0.8], start=1)]
    assert steps_to(curve, 0.5) == 750 and steps_to(curve, 0.9) == 1000 and steps_to(curve, 0.99) is None
    # First cell earlier in 4 of 5 seeds by >= 1,000 updates at the median: favors first.
    result = compare([2000, 3000, 4000, 5000, 9000], [4000, 5000, 6000, 7000, 8000], 10000, 250)
    assert result['first_earlier_seeds'] == 4 and result['median_difference'] == 2000 and result['favors_first']
    # A censored second cell ranks after every reached value; both censored is a tie.
    result = compare([9000, None, 1000, 1000, 1000], [None, None, 3000, 3000, 3000], 10000, 250)
    assert result['differences'] == [1250, 0, 2000, 2000, 2000] and result['favors_first']
    # Earlier in only 3 seeds does not favor either cell.
    assert not compare([1000, 1000, 1000, 9000, 9000], [5000, 5000, 5000, 5000, 5000], 10000, 250)['favors_first']

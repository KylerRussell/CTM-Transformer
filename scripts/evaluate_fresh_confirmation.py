"""Evaluate locked recipes on paired fresh maps after every checkpoint is frozen."""
import argparse
import gc
import json
from pathlib import Path
import torch
from ctm_transformer.experiment import file_hash
from ctm_transformer.algorithmic import load_algorithmic_split
from ctm_transformer.fresh_pointer import validate_fresh_suite
from ctm_transformer.readout_selection import evaluate_readouts
from scripts.eval_harness import _load_checkpoint
from scripts.confirmation_common import read

ROOT = Path('research/results/fresh_confirmation_v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cells', nargs='+')
    args = parser.parse_args()
    frozen = read(ROOT / 'checkpoints.json')
    assert frozen['complete'] and frozen['no_fresh_test_forward_before_freeze']
    for path, digest in frozen['source_sha256'].items():
        assert file_hash(path) == digest, path
    assert file_hash(ROOT / 'recipes.json') == frozen['recipes_sha256']
    assert file_hash(ROOT / 'registry.json') == frozen['registry_sha256']
    assert file_hash(Path(frozen['fresh_dataset']) / 'manifest.json') == frozen['fresh_manifest_sha256']
    validate_fresh_suite(frozen['fresh_dataset'])
    by_cell = {record['cell']: record for record in frozen['models']}
    cells = args.cells or list(by_cell)
    assert len(cells) == len(set(cells)) and set(cells) <= set(by_cell)
    # Fail before any forward if a requested output already exists.
    assert all(not (ROOT / f'{cell}.evaluation.json').exists() for cell in cells)
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    for cell in cells:
        record = by_cell[cell]
        assert file_hash(record['checkpoint']) == record['checkpoint_sha256']
        model, config = _load_checkpoint(record['checkpoint'])
        model.to(args.device).eval()
        results = {}
        for split in ('validation', 'test_id', 'test_shuffled'):
            root = 'research/data/ordered_pointer_v1' if split == 'validation' else frozen['fresh_dataset']
            dataset = load_algorithmic_split(root, 'pointer', split, config.seq_len)
            values = evaluate_readouts(model, dataset, config, args.device, (record['readout'],))
            metrics = values[record['readout']]
            expected_count = 128 if split == 'validation' else 512
            assert len(metrics['predictions']) == expected_count
            assert metrics['target_tokens'] == 2 * expected_count
            if split == 'validation':
                # Check exact generation and numerical CE agreement before fresh-test forwards.
                assert abs(metrics['loss'] - record['validation_ce']) < 1e-6
                assert metrics['generation'] == record['validation_generation']
            results[split] = {'dataset': dataset.metadata, 'metrics': metrics,
                              'diagnostics': values['diagnostics']}
        report = {
            'complete': True, 'cell': cell, 'role': record['role'], 'model_family': record['model_family'],
            'seed': record['seed'], 'selected': record, 'checkpoint': model._checkpoint_metadata,
            'checkpoint_freeze_sha256': file_hash(ROOT / 'checkpoints.json'),
            'fresh_manifest_sha256': frozen['fresh_manifest_sha256'],
            'source_sha256': frozen['source_sha256'], 'device': args.device,
            'gpu': torch.cuda.get_device_name(args.device), 'results': results,
            'scope': 'Fixed-policy teacher-forced answer/EOS CE and unrestricted exact answer+EOS generation; paired 512 fresh semantic maps. No latency benchmark.'}
        with (ROOT / f'{cell}.evaluation.json').open('x') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        print(cell, {split: results[split]['metrics']['generation'] for split in ('test_id', 'test_shuffled')}, flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

"""Behavioral checks for research baselines and the shared experiment interface."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ctm_transformer.baselines import BaselineConfig, BaselineTransformer
from ctm_transformer.experiment import evaluate_loss, load_text_blocks, training_depth, train_experiment
from ctm_transformer.research import build_model, load_research_config, parameter_counts

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['transformer', 'recurrent_depth']


@pytest.fixture
def device():
    value = os.environ.get('CTM_TEST_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.empty(1, device=value)
    torch.set_num_threads(4)
    return value


def small(family, **changes):
    values = dict(model_family=family, vocab_size=32, d_model=32, n_heads=4,
                  ffn_hidden_dim=64, max_seq_len=16, seq_len=8, batch_size=2,
                  n_layers=2, max_thought_steps=1, train_depth_min=1, train_depth_max=1,
                  dtype='float32', bf16_autocast=False)
    if family == 'recurrent_depth':
        values.update(n_layers=0, prelude_layers=1, core_layers=2, coda_layers=1,
                      max_thought_steps=3, train_depth_min=3, train_depth_max=3)
    return BaselineConfig(**(values | changes))


@pytest.mark.parametrize('family', FAMILIES)
def test_causality_batching_and_final_ce(device, family):
    torch.manual_seed(71)
    model = build_model(small(family)).to(device).eval()
    ids = torch.randint(0, 32, (2, 8), device=device)
    changed = ids.clone(); changed[:, 4:] = (changed[:, 4:] + 7) % 32
    targets = (ids + 1) % 32; targets[:, :3] = -100
    with torch.no_grad():
        result = model(ids, targets)
        torch.testing.assert_close(result['logits'][:, :4], model(changed)['logits'][:, :4])
        torch.testing.assert_close(result['logits'][0:1, :5], model(ids[:1, :5])['logits'])
        torch.testing.assert_close(result['loss'], F.cross_entropy(result['logits'].reshape(-1, 32), targets.reshape(-1)))
        permuted = ids[:, [2, 0, 1, 3, 4, 5, 6, 7]]
        assert not torch.allclose(result['logits'][:, -1], model(permuted)['logits'][:, -1], atol=1e-6)


def test_recurrent_core_is_shared_and_injected_each_tick(device):
    torch.manual_seed(72)
    config = small('recurrent_depth')
    model = build_model(config).to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    calls, core_calls = [], []
    handle = model.injection.register_forward_pre_hook(lambda module, args: calls.append(args[0].detach()))
    h2 = model.core[0].register_forward_hook(lambda *args: core_calls.append(1))
    result = model(ids, max_thought_steps=7)
    handle.remove(); h2.remove()
    assert len(calls) == len(core_calls) == 7
    assert result['block_applications'] == 16
    assert torch.count_nonzero(calls[0][..., :32]) == 0
    for call in calls:
        torch.testing.assert_close(call[..., 32:], calls[0][..., 32:])
    # Compare to an independently written explicit unroll using the same modules.
    embedded = model.token_embedding(ids) + model.pos_embedding(torch.arange(4, device=device))
    for layer in model.prelude:
        embedded = layer(embedded)
    state = torch.zeros_like(embedded)
    for _ in range(7):
        state = model.injection(torch.cat((state, embedded), -1))
        for layer in model.core:
            state = layer(state)
    state = model.final_norm(state)
    for layer in model.coda:
        state = layer(state)
    torch.testing.assert_close(result['logits'], model.lm_head(model.final_norm(state)))
    result['logits'].square().mean().backward()
    assert model.prelude[0].attn.v.weight.grad.abs().sum() > 0
    assert model.core[0].attn.v.weight.grad.abs().sum() > 0
    assert parameter_counts(model)['total'] == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('dropout', [0.0, 0.1])
def test_checkpoint_recomputation_gradients(device, family, dropout):
    torch.manual_seed(73)
    reference = build_model(small(family, dropout=dropout)).to(device).train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.config.gradient_checkpointing = True
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    torch.manual_seed(74); loss1 = reference(ids, ids + 1)['loss']; loss1.backward()
    torch.manual_seed(74); loss2 = checkpointed(ids, ids + 1)['loss']; loss2.backward()
    torch.testing.assert_close(loss1, loss2)
    for (name, p), (_, q) in zip(reference.named_parameters(), checkpointed.named_parameters()):
        assert p.grad is not None and q.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


@pytest.mark.parametrize('family', FAMILIES)
def test_tiny_batch_learning(device, family):
    torch.manual_seed(75)
    model = build_model(small(family)).to(device).train()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]], device=device)
    targets = ids + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = float(model(ids, targets)['loss'].detach())
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, targets)['loss']; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    final = float(model(ids, targets)['loss'].detach())
    assert final < min(0.5, initial * 0.2), (initial, final)


@pytest.mark.parametrize('family', FAMILIES)
def test_strict_roundtrip_and_likelihood_adapter(device, family, tmp_path):
    from scripts.eval_harness import _load_checkpoint, CTMTransformerLM
    model = build_model(small(family)).to(device).eval()
    path = tmp_path / 'model.pt'
    saved = {'model_family': family, 'config': asdict(model.config), 'model_state_dict': model.state_dict()}
    torch.save(saved, path)
    restored, config = _load_checkpoint(path); restored.to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    with torch.no_grad():
        torch.testing.assert_close(restored(ids)['logits'], model(ids)['logits'])
    class Tokenizer:
        n_vocab = 32
        eot_token = 0
        def encode(self, s):
            return [ord(c) - ord('a') + 1 for c in s]
        def decode(self, tokens):
            return ''.join(chr(t + ord('a') - 1) for t in tokens)
    lm = CTMTransformerLM(restored, Tokenizer(), device=device, max_seq_len=16)
    score, _ = lm.loglikelihood([('ab', 'cd')])[0]
    with torch.no_grad():
        logits = model(ids[:, :3])['logits'].float().log_softmax(-1)
        expected = float(logits[0, 1, 3] + logits[0, 2, 4])
    assert score == pytest.approx(expected, abs=1e-5)
    del saved['model_state_dict']['lm_head.weight']; torch.save(saved, path)
    with pytest.raises(RuntimeError):
        _load_checkpoint(path)


def test_fixed_transformer_rejects_depth_sweeps():
    model = build_model(small('transformer'))
    with pytest.raises(ValueError, match='fixed'):
        model(torch.tensor([[1, 2]]), max_thought_steps=2)


def test_variable_depth_is_seeded_and_isolated():
    config = small('recurrent_depth', train_depth_min=1, train_depth_max=8)
    def draw():
        rng = torch.Generator().manual_seed(17)
        return [training_depth(config, rng) for _ in range(80)]
    expected = draw(); torch.randn(1000)
    assert draw() == expected
    assert set(expected) == set(range(1, 9))


def test_text_windows_and_split_fingerprint(tmp_path):
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text)))
    path = tmp_path / 'data.txt'; path.write_text('abcdefghijkl')
    blocks, meta = load_text_blocks(path, Tokenizer(), 4)
    assert blocks.tolist() == [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]
    assert meta['unused_input_tokens'] == 3


@pytest.mark.parametrize('family', FAMILIES)
def test_validation_includes_partial_batch_token_weighting(device, family):
    config = small(family)
    model = build_model(config).to(device).train()
    blocks = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    actual = evaluate_loss(model, blocks, config, device)
    assert model.training and actual['target_tokens'] == 9
    with torch.no_grad():
        expected = model(blocks[:, :-1].to(device), blocks[:, 1:].to(device))['loss'].item()
    assert actual['loss'] == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize('family', ['ctm', *FAMILIES])
def test_shared_runner_trains_evaluates_and_saves(device, family, tmp_path):
    from scripts.eval_harness import _load_checkpoint
    if torch.device(device).type != 'cuda':
        pytest.skip('Shared runner intentionally requires CUDA')
    train_path, eval_path = tmp_path / 'train.txt', tmp_path / 'eval.txt'
    train_path.write_text('The quick brown fox jumps over the lazy dog. ' * 12)
    eval_path.write_text('A small bird flies across the quiet garden. ' * 7)
    if family == 'ctm':
        config, identity = load_research_config(ROOT / 'research/configs/ctm_reference_v1.json')
        config = replace(config, d_model=16, d_latent=16, n_heads=2, n_layers=1, nlm_groups=16,
                         nlm_hidden_dim=4, history_len=4, sync_sparse_pairs=16, max_thought_steps=2)
    else:
        config = small(family, vocab_size=50257, dtype='bfloat16', bf16_autocast=True)
        identity = {'name': 'test-fixture', 'sha256': None}
    config = replace(config, batch_size=2, seq_len=8, max_seq_len=16, max_steps=3, warmup_steps=0,
                     eval_interval=2, log_interval=3, data_path=str(train_path), eval_data_path=str(eval_path),
                     checkpoint_dir=str(tmp_path / 'run'), device=device, gradient_accumulation_steps=2)
    summary = train_experiment(config, identity, 17)
    assert summary['complete'] and summary['tokens_seen'] == 96
    loaded, loaded_config = _load_checkpoint(tmp_path / 'run/final.pt')
    assert asdict(loaded_config) == asdict(config)
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in loaded.parameters())
    rows = [json.loads(line) for line in (tmp_path / 'run/metrics.jsonl').read_text().splitlines()]
    assert 'validation' in rows[1] and 'validation' in rows[2]
    assert (tmp_path / 'run/tokenizer.json').is_file()
    with pytest.raises(ValueError, match='empty checkpoint_dir'):
        train_experiment(config, identity, 17)

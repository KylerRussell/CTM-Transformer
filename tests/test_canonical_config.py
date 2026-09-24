"""Contract and objective tests for the versioned CTM research reference."""
import copy
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from ctm_transformer.research import load_research_config

PRESET = Path(__file__).resolve().parents[1] / 'research/configs/ctm_reference_v1.json'


@pytest.fixture
def device():
    value = os.environ.get('CTM_TEST_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.empty(1, device=value)
    return value


def small_config(**changes):
    config, _ = load_research_config(PRESET)
    return replace(config, **(dict(vocab_size=32, d_model=16, d_latent=16, nlm_groups=16,
        n_heads=2, n_layers=2, nlm_hidden_dim=4, history_len=4, sync_sparse_pairs=16,
        max_thought_steps=3, seq_len=4, max_seq_len=16, batch_size=2) | changes))


def test_frozen_reference_contract():
    config, identity = load_research_config(PRESET)
    assert identity['name'] == 'ctm_reference_v1'
    assert config.bf16_autocast and config.dtype == 'bfloat16'
    assert config.nlm_groups == config.d_latent == 256
    assert config.sync_method == 'sparse_decay'
    assert config.use_positional_encoding and not config.use_attention_residuals
    assert config.temporal_loss_type == 'final_ce' and config.mono_penalty_weight == 0
    assert not any((config.use_shared_head_film, config.per_tick_heads, config.use_loop_pos_emb,
                    config.use_distillation, config.use_feec, config.use_matrix_streams,
                    config.use_engram, config.use_hyperloop, config.use_two_phase_curriculum))


def test_partial_presets_are_rejected(tmp_path):
    saved = json.loads(PRESET.read_text())
    del saved['config']['use_attention_residuals']
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match='fully specified'):
        load_research_config(path)


def test_final_ce_requires_no_monotonic_penalty():
    with pytest.raises(ValueError, match='mono_penalty_weight=0'):
        CTMTransformer(small_config(mono_penalty_weight=0.5))


@pytest.mark.parametrize('masked', [False, True])
def test_loss_is_exactly_final_tick_ce(device, masked):
    torch.manual_seed(41)
    model = CTMTransformer(small_config()).to(device).eval()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    targets = ids + 1
    if masked:
        targets[:, :-1] = -100
    result = model(ids, targets=targets)
    expected = F.cross_entropy(result['logits'].reshape(-1, 32), targets.reshape(-1))
    torch.testing.assert_close(result['loss'], expected)
    gradients = torch.autograd.grad(result['loss'], result['all_logits'], retain_graph=True)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in gradients[:-1])
    assert torch.count_nonzero(gradients[-1]) > 0
    result['loss'].backward()
    assert model.layers[0].nlm.w1.grad.abs().sum() > 0


def test_sequential_layers_have_no_attention_mixing(device):
    model = CTMTransformer(small_config()).to(device).eval()
    assert not any(name.startswith('attn_res_') for name, _ in model.named_parameters())
    inputs, outputs = [], []
    for layer in model.layers:
        layer.register_forward_pre_hook(lambda module, args: inputs.append(args[2]))
        layer.register_forward_hook(lambda module, args, output: outputs.append(output[0]))
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]], device=device))
    for offset in range(0, len(inputs), model.config.n_layers):
        torch.testing.assert_close(inputs[offset + 1], outputs[offset])
    legacy = CTMTransformer(small_config(use_attention_residuals=True))
    assert any(name.startswith('attn_res_') for name, _ in legacy.named_parameters())


@pytest.mark.parametrize('dropout', [0.0, 0.1])
def test_canonical_checkpoint_gradients(device, dropout):
    torch.manual_seed(42)
    reference = CTMTransformer(small_config(dropout=dropout)).to(device).train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.config.gradient_checkpointing = True
    checkpointed.config.gradient_checkpointing_min_T = 1
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    torch.manual_seed(43)
    expected = reference(ids, targets=ids + 1)['loss']
    expected.backward()
    torch.manual_seed(43)
    actual = checkpointed(ids, targets=ids + 1)['loss']
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for (name, parameter), (_, other) in zip(reference.named_parameters(), checkpointed.named_parameters()):
        assert (parameter.grad is None) == (other.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_canonical_roundtrip_and_unseen_thought_depth(device, tmp_path):
    from scripts.eval_harness import _load_checkpoint
    model = CTMTransformer(small_config()).to(device).eval()
    ids = torch.tensor([[1, 2, 3]], device=device)
    with torch.no_grad():
        expected = model(ids, max_thought_steps=6)['logits']
    path = tmp_path / 'reference.pt'
    torch.save({'config': asdict(model.config), 'model_state_dict': model.state_dict()}, path)
    loaded, config = _load_checkpoint(path)
    loaded.to(device).eval()
    assert not config.use_attention_residuals and config.temporal_loss_type == 'final_ce'
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids, max_thought_steps=6)['logits'], expected)


def test_preexisting_checkpoint_retains_attention_residuals(tmp_path):
    from scripts.eval_harness import _load_checkpoint
    config = replace(small_config(), use_attention_residuals=True, temporal_loss_type='ramp_mono')
    model = CTMTransformer(config)
    saved = asdict(config)
    del saved['use_attention_residuals']
    del saved['bf16_autocast']
    path = tmp_path / 'legacy.pt'
    torch.save({'config': saved, 'model_state_dict': model.state_dict()}, path)
    restored, restored_config = _load_checkpoint(path)
    assert restored_config.use_attention_residuals
    assert not restored_config.bf16_autocast
    assert set(restored.state_dict()) == set(model.state_dict())


def test_launcher_dry_run_preserves_architecture(tmp_path, monkeypatch, capsys):
    from scripts.train_research import main
    training, evaluation = tmp_path / 'train.txt', tmp_path / 'eval.txt'
    training.write_text('training fixture')
    evaluation.write_text('evaluation fixture')
    monkeypatch.setattr(sys, 'argv', ['train_research', '--config', str(PRESET),
        '--data_path', str(training), '--eval_data_path', str(evaluation),
        '--checkpoint_dir', str(tmp_path / 'run'), '--max_steps', '150', '--dry_run'])
    main()
    record = json.loads(capsys.readouterr().out)
    expected, _ = load_research_config(PRESET)
    expected = replace(expected, device='cuda:0', data_path=str(training),
                       eval_data_path=str(evaluation), checkpoint_dir=str(tmp_path / 'run'), max_steps=150)
    assert record['effective_config'] == json.loads(json.dumps(asdict(expected)))
    assert not (tmp_path / 'run').exists()

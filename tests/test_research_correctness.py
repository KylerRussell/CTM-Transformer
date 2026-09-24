"""Research gates on the actual model; select a GPU with CTM_TEST_DEVICE=cuda:0."""

import copy
import os

import pytest
import torch

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer, NeuronLevelModels


@pytest.fixture
def device():
    # Explicit CUDA requests must fail if CUDA is unavailable, never silently
    # turn a requested GPU check into a CPU result.
    selected = os.environ.get("CTM_TEST_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    torch.empty(1, device=selected)
    return selected


def small_config(**overrides):
    values = dict(
        vocab_size=32, d_model=16, d_latent=16, n_heads=2, n_layers=2,
        history_len=4, nlm_hidden_dim=8, max_thought_steps=3,
        max_seq_len=16, dropout=0.0, gradient_checkpointing=False,
        use_positional_encoding=True,
    )
    return CTMConfig(**(values | overrides))


def make_model(device, **overrides):
    torch.manual_seed(17)
    return CTMTransformer(small_config(**overrides)).to(device)


@torch.no_grad()
def test_future_tokens_and_padding_do_not_change_prefix(device):
    model = make_model(device).eval()
    prefix = torch.tensor([[1, 2, 3, 4]], device=device)
    expected = model(prefix)["logits"]
    extended = torch.tensor([[1, 2, 3, 4, 7, 8], [1, 2, 3, 4, 0, 0]], device=device)
    mask = torch.tensor([[False] * 6, [False] * 4 + [True] * 2], device=device)
    actual = model(extended, key_padding_mask=mask)["logits"][:, :4]
    torch.testing.assert_close(actual, expected.expand(2, -1, -1), atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("positions", [False, True])
@torch.no_grad()
def test_prefix_order_sensitivity(device, positions):
    model = make_model(device, use_positional_encoding=positions).eval()
    # Same prefix multiset and same final token, different prefix order.
    ids = torch.tensor([[1, 2, 3, 4], [3, 1, 2, 4]], device=device)
    logits = model(ids)["logits"][:, -1]
    if positions:
        assert (logits[0] - logits[1]).abs().max().item() > 1e-5
    else:
        torch.testing.assert_close(logits[0], logits[1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("groups", [1, 4, 16])
def test_temporal_mlp_sharing_and_gradients(device, groups):
    nlm = NeuronLevelModels(16, 4, 8, nlm_groups=groups, dropout=0).to(device)
    # All neurons see identical histories. Neurons within a group must share
    # a temporal response, while independent groups can learn different ones.
    history = torch.randn(2, 4, 1, device=device).expand(-1, -1, 16)
    out = nlm(history).reshape(2, groups, 16 // groups)
    torch.testing.assert_close(out, out[:, :, :1].expand_as(out))
    out.sum().backward()
    assert nlm.w1.grad is not None and torch.isfinite(nlm.w1.grad).all()
    assert nlm.w1.shape[0] == groups


def test_sparse_pair_configuration_reaches_every_layer(device):
    model = make_model(device, sync_method="sparse_decay", sync_sparse_pairs=11)
    assert all(layer.sync_computer.output_dim == 11 for layer in model.layers)
    assert model(torch.tensor([[1, 2, 3]], device=device))["logits"].shape == (1, 3, 32)


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("variant", ["standard", "hyperloop", "matrix", "loop_embeddings"])
def test_checkpointing_preserves_loss_and_all_gradients(device, dropout, variant):
    options = {
        "standard": {},
        "hyperloop": dict(use_hyperloop=True, hyperloop_n_begin=1,
                          hyperloop_n_middle=1, hyperloop_n_end=1, hyperloop_middle_loops=2),
        "matrix": dict(use_matrix_streams=True),
        "loop_embeddings": dict(use_loop_pos_emb=True),
    }[variant]
    reference = make_model(device, dropout=dropout, **options).train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.config = copy.deepcopy(reference.config)
    checkpointed.config.gradient_checkpointing = True
    checkpointed.config.gradient_checkpointing_min_T = 1
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    targets = (ids + 1) % 32
    torch.manual_seed(123)
    loss = reference(ids, targets=targets)["loss"]
    loss.backward()
    torch.manual_seed(123)
    ckpt_loss = checkpointed(ids, targets=targets)["loss"]
    ckpt_loss.backward()
    torch.testing.assert_close(ckpt_loss, loss, atol=2e-6, rtol=2e-5)
    for (name, param), (other_name, other) in zip(
        reference.named_parameters(), checkpointed.named_parameters()
    ):
        assert name == other_name
        assert (param.grad is None) == (other.grad is None), name
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_loop_embeddings_affect_loss_and_receive_gradients(device):
    model = make_model(device, use_loop_pos_emb=True).train()
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    loss = model(ids, targets=ids + 1)["loss"]
    loss.backward()
    grad = model.loop_pos_emb.weight.grad
    assert grad is not None
    assert torch.all(grad.abs().sum(dim=1) > 0)


@pytest.mark.parametrize("steps", [0, -1])
def test_nonpositive_thought_budget_is_rejected(device, steps):
    model = make_model(device).eval()
    with pytest.raises(ValueError, match="positive"):
        model(torch.tensor([[1, 2]], device=device), max_thought_steps=steps)


@pytest.mark.parametrize("option", ["per_tick_heads", "use_shared_head_film", "use_loop_pos_emb"])
def test_indexed_tick_parameters_reject_untrained_depth(device, option):
    model = make_model(device, **{option: True}).eval()
    with pytest.raises(ValueError, match="max_thought_steps"):
        model(torch.tensor([[1, 2]], device=device), max_thought_steps=4)


@torch.no_grad()
def test_shared_head_supports_depth_extrapolation_and_state_roundtrip(device, tmp_path):
    model = make_model(device).eval()
    ids = torch.tensor([[1, 2, 3]], device=device)
    expected = model(ids, max_thought_steps=5)["logits"]
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = make_model(device).eval()
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    torch.testing.assert_close(restored(ids, max_thought_steps=5)["logits"], expected)


def test_tiny_batch_can_be_learned(device):
    model = make_model(device).train()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    targets = ids + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    initial = float(model(ids, targets=targets)["loss"].detach())
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, targets=targets)["loss"]
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(ids)["logits"]
        final = torch.nn.functional.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    assert final.item() < initial * 0.2
    assert (logits.argmax(-1) == targets).float().mean().item() >= 0.95

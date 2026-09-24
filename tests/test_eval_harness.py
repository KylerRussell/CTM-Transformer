"""Independent likelihood, tokenizer-boundary, loading, and provenance gates."""
from dataclasses import asdict
import json
import math
import os
from types import SimpleNamespace

import pytest
import tiktoken
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM

from ctm_transformer.config import CTMConfig
from ctm_transformer.model import CTMTransformer
from scripts.eval_harness import (
    CTMTransformerLM, _load_checkpoint, _tokenizer_snapshot, _write_json,
    _dataset_metadata, _code_metadata, _sha256_file,
)


class DigitTokenizer:
    eot_token = 0
    n_vocab = 16

    def encode(self, text):
        return [int(char) + 1 for char in text]

    def decode(self, tokens):
        return ''.join(str(token - 1) for token in tokens)


class TransitionModel(torch.nn.Module):
    """Known bigram distribution: row i favors token (i+1) modulo V."""
    def __init__(self, device):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16, max_seq_len=8)
        rows = torch.arange(16)[:, None]
        columns = torch.arange(16)[None, :]
        self.register_buffer('scores', -((columns - rows - 1) % 16).float().to(device))
        self.calls = []

    def forward(self, ids, max_thought_steps=None):
        self.calls.append(ids.detach().cpu())
        return {'logits': self.scores[ids]}


@pytest.fixture
def device():
    selected = os.environ.get('CTM_TEST_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.empty(1, device=selected)
    return selected


@pytest.fixture
def bpe():
    # Real BPE algorithm, offline and tiny. Includes whitespace-sensitive merges
    # and a merge spanning a deliberately invalid context boundary.
    return tiktoken.Encoding(name='research-test-bpe', pat_str=r'\s?[^\s]+|\s+',
                            mergeable_ranks={**{bytes([i]): i for i in range(256)},
                                             b' b': 256, b'ab': 257},
                            special_tokens={'<|endoftext|>': 258})


def config(**overrides):
    fields = dict(vocab_size=259, d_model=16, d_latent=16, n_heads=2, n_layers=2,
                  max_seq_len=8, seq_len=8, history_len=3, nlm_hidden_dim=4,
                  max_thought_steps=2, use_positional_encoding=True,
                  gradient_checkpointing=False, dropout=0.0)
    return CTMConfig(**(fields | overrides))


def instance(pair):
    return Instance(request_type='loglikelihood', doc={}, arguments=pair, idx=0)


@pytest.mark.parametrize('pair', [('', '0'), ('', '012'), ('01', '23'),
                                 ('543210', '1234'), ('123', '0123'), ('1', '111'), ('', '')])
def test_analytic_scores_and_greedy_flags(device, pair):
    tokenizer = DigitTokenizer()
    model = TransitionModel(device)
    adapter = CTMTransformerLM(model, tokenizer, device=device, max_seq_len=4)
    result = adapter.loglikelihood([instance(pair)])[0]
    ctx, cont = tokenizer.encode(pair[0]) or [0], tokenizer.encode(pair[1])
    # Closed-form bigram likelihood; independent of batching, model forward,
    # truncation, or the adapter's tensor indexing.
    normalizer = math.log(sum(math.exp(-j) for j in range(16)))
    previous = ctx[-1]
    expected = 0.0
    greedy = True
    for token in cont:
        offset = (token - previous - 1) % 16
        expected += -offset - normalizer
        greedy = greedy and offset == 0
        previous = token
    assert result[0] == pytest.approx(expected, abs=3e-6)
    assert result[1] == greedy
    assert all(call.shape[1] <= 4 for call in model.calls)


def test_continuation_is_never_silently_truncated(device):
    model = TransitionModel(device)
    adapter = CTMTransformerLM(model, DigitTokenizer(), device=device, max_seq_len=4)
    with pytest.raises(ValueError, match='discard scored tokens'):
        adapter.loglikelihood([('', '01234')])
    assert not model.calls


def test_empty_request_list(device):
    model = TransitionModel(device)
    adapter = CTMTransformerLM(model, DigitTokenizer(), device=device, max_seq_len=4)
    assert adapter.loglikelihood([]) == []
    assert not model.calls


def test_bpe_whitespace_boundary_and_empty_context(device, bpe):
    adapter = CTMTransformerLM(CTMTransformer(config()).to(device), bpe,
                               device=device, max_seq_len=8)
    assert adapter._encode_pair('a ', 'b') == ([97], [256])
    assert adapter._encode_pair('a ', 'b') == adapter._encode_pair('a', ' b')
    assert adapter._encode_pair('', 'a') == ([258], [97])
    assert adapter._encode_pair(' ', 'b') == ([258], [256])
    with pytest.raises(ValueError, match='merged token'):
        adapter._encode_pair('a', 'b')


def test_explicit_prefix_required_if_tokenizer_has_none(device):
    tokenizer = DigitTokenizer()
    tokenizer.eot_token = None
    with pytest.raises(ValueError, match='prefix_token_id'):
        CTMTransformerLM(TransitionModel(device), tokenizer, device=device, max_seq_len=4)
    adapter = CTMTransformerLM(TransitionModel(device), tokenizer, device=device,
                               max_seq_len=4, prefix_token_id=1)
    assert adapter.prefix_token_id == 1


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_ctm_matches_token_by_token_scores_and_batching(device, bpe, dtype):
    torch.manual_seed(31)
    model = CTMTransformer(config()).to(device=device, dtype=dtype).eval()
    adapter = CTMTransformerLM(model, bpe, device=device, batch_size=3, max_seq_len=8)
    # Explicit token lists are independent of the adapter's pair preparation.
    cases = [(('', 'a'), [258], [97]), (('a ', 'b'), [97], [256]),
             (('123456789', '0123'), list(b'123456789'), list(b'0123')),
             (('a', ' xyz'), [97], list(b' xyz'))]
    actual = adapter.loglikelihood([pair for pair, _, _ in cases])
    for result, (_, ctx, cont) in zip(actual, cases):
        # Match the retained context used by the declared full-continuation
        # window policy, then independently run one prefix per target token.
        retained = ctx[-(adapter.max_length + 1 - len(cont)):]
        expected = 0.0
        greedy = True
        for token in cont:
            ids = torch.tensor([retained], device=device)
            with torch.no_grad():
                logits = model(ids)['logits'][0, -1].float()
                expected += (logits[token] - torch.logsumexp(logits, 0)).item()
                greedy = greedy and logits.argmax().item() == token
            retained.append(token)
        tolerance = 0.025 if dtype == torch.bfloat16 else 2e-5
        assert result[0] == pytest.approx(expected, abs=tolerance)
        assert result[1] == greedy
    adapter._batch_size = 1
    individual = adapter.loglikelihood([pair for pair, _, _ in cases])
    assert [r[0] for r in actual] == pytest.approx([r[0] for r in individual], abs=tolerance)
    assert [r[1] for r in actual] == [r[1] for r in individual]


def test_matches_pinned_harness_reference(device, bpe):
    from lm_eval.models.huggingface import HFLM

    class Reference(CTMTransformerLM):
        # Use the upstream algorithms directly on the same model. No pretrained
        # HF model or weights are needed to compare scoring implementations.
        backend = 'causal'
        logits_cache = False
        softmax_dtype = torch.float32
        _encode_pair = TemplateLM._encode_pair
        loglikelihood = TemplateLM.loglikelihood
        _loglikelihood_tokens = HFLM._loglikelihood_tokens
        _select_cont_toks = HFLM._select_cont_toks

    torch.manual_seed(32)
    model = CTMTransformer(config()).to(device).eval()
    ours = CTMTransformerLM(model, bpe, device=device, batch_size=3, max_seq_len=8)
    reference = Reference(model, bpe, device=device, batch_size=3, max_seq_len=8)
    requests = [instance(pair) for pair in [('', 'a'), ('a ', 'b'),
                ('123456789', '0123'), ('a', ' xyz')]]
    expected = reference.loglikelihood(requests, disable_tqdm=True)
    actual = ours.loglikelihood(requests)
    assert [r[0] for r in actual] == pytest.approx([r[0] for r in expected], abs=2e-5)
    assert [r[1] for r in actual] == [r[1] for r in expected]


@pytest.mark.parametrize('prefix', ['', 'module._orig_mod.', '_orig_mod.module.', 'module.module.'])
def test_strict_checkpoint_roundtrip(tmp_path, device, prefix):
    torch.manual_seed(11)
    original = CTMTransformer(config()).to(device).eval()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'config': asdict(original.config), 'step': 42,
                'model_state_dict': {prefix + k: v for k, v in original.state_dict().items()}}, path)
    loaded, loaded_config = _load_checkpoint(path)
    loaded.to(device).eval()
    ids = torch.tensor([[1, 2, 3]], device=device)
    with torch.no_grad():
        torch.testing.assert_close(original(ids)['logits'], loaded(ids)['logits'])
    assert asdict(loaded_config) == asdict(original.config)
    assert loaded._checkpoint_metadata['training_counters'] == {'step': 42}
    assert loaded._checkpoint_metadata['sha256'] == _sha256_file(path)


@pytest.mark.parametrize('problem', ['missing', 'unexpected', 'collision', 'unknown_config', 'tied'])
def test_invalid_checkpoints_are_rejected(tmp_path, problem):
    cfg = config(use_shared_head_film=True, tie_embeddings=True)
    model = CTMTransformer(cfg)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    saved_config = asdict(cfg)
    if problem == 'missing':
        del state['z0']
    elif problem == 'unexpected':
        state['unknown.weight'] = torch.zeros(1)
    elif problem == 'collision':
        state['module.z0'] = state['z0']
    elif problem == 'unknown_config':
        saved_config['unknown_feature'] = True
    elif problem == 'tied':
        state['lm_head.weight'].add_(1)
    path = tmp_path / 'bad.pt'
    torch.save({'config': saved_config, 'model': state}, path)
    with pytest.raises((RuntimeError, ValueError)):
        _load_checkpoint(path)


def test_tokenizer_snapshot_roundtrip(bpe, tmp_path):
    snapshot = _tokenizer_snapshot(bpe)
    import base64
    restored = tiktoken.Encoding(name=snapshot['name'], pat_str=snapshot['pattern'],
        mergeable_ranks={base64.b64decode(token): rank for token, rank in snapshot['mergeable_ranks']},
        special_tokens=snapshot['special_tokens'])
    for text in ['a b', 'ab', '  b', '123', '']:
        assert restored.encode(text) == bpe.encode(text)
    path = tmp_path / 'tokenizer.json'
    _write_json(path, snapshot)
    assert json.loads(path.read_text()) == snapshot
    assert len(_sha256_file(path)) == 64


def test_dataset_and_code_provenance():
    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_dict({'text': ['one', 'two']})
    task = SimpleNamespace(dataset=DatasetDict(test=dataset),
                           dump_config=lambda: {'dataset_path': 'local_fixture',
                                                'dataset_kwargs': {'revision': 'fixed-test-revision'}})
    info = _dataset_metadata({'group': {'fixture': task}})['fixture']
    assert info['splits']['test']['fingerprint'] == dataset._fingerprint
    assert info['splits']['test']['num_rows'] == 2
    assert info['dataset_kwargs']['revision'] == 'fixed-test-revision'
    code = _code_metadata()
    assert 'scripts/eval_harness.py' in code['source_sha256']
    assert 'ctm_transformer/model.py' in code['source_sha256']


def test_hf_fast_tokenizer_snapshot_roundtrip():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from ctm_transformer.train import HFTokenizerWrapper
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[EOS]': 1, 'hello': 2, 'world': 3}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='[EOS]', unk_token='[UNK]')
    snapshot = _tokenizer_snapshot(HFTokenizerWrapper(tokenizer))
    restored = Tokenizer.from_str(json.dumps(snapshot['backend']))
    assert restored.encode('hello world').ids == tokenizer.encode('hello world', add_special_tokens=False)
    assert snapshot['special_tokens_map']['eos_token'] == '[EOS]'


def test_evaluation_command_records_replayable_results(tmp_path, device, bpe, monkeypatch):
    import sys
    import yaml
    from scripts.eval_harness import main

    checkpoint = tmp_path / 'tiny.pt'
    torch.manual_seed(8)
    model = CTMTransformer(config())
    torch.save({'config': asdict(model.config), 'model_state_dict': model.state_dict(), 'step': 0}, checkpoint)
    documents = [{'query': 'a', 'choices': ['b', 'xyz'], 'answer': 0},
                 {'query': 'Question123', 'choices': ['a', 'b'], 'answer': 1}]
    dataset_path = tmp_path / 'samples.jsonl'
    dataset_path.write_text(''.join(json.dumps(doc) + '\n' for doc in documents))
    task_dir = tmp_path / 'tasks'
    task_dir.mkdir()
    task = {'task': 'ctm_fixture', 'dataset_path': 'json',
            'dataset_kwargs': {'data_files': {'test': str(dataset_path)}},
            'test_split': 'test', 'output_type': 'multiple_choice',
            'doc_to_text': '{{query}}', 'doc_to_choice': '{{choices}}',
            'doc_to_target': '{{answer}}', 'num_fewshot': 0,
            'metric_list': [{'metric': 'acc', 'aggregation': 'mean', 'higher_is_better': True}],
            'metadata': {'version': 1}}
    (task_dir / 'fixture.yaml').write_text(yaml.safe_dump(task))
    output = tmp_path / 'eval.json'
    monkeypatch.setattr(tiktoken, 'get_encoding', lambda name: bpe)
    monkeypatch.setattr(sys, 'argv', ['eval_harness', '--checkpoint', str(checkpoint),
        '--tasks', 'ctm_fixture', '--include_path', str(task_dir), '--device', device,
        '--dtype', 'float32', '--t_sweep', '1,2', '--limit', '2', '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    assert report['schema_version'] == 2 and report['complete']
    assert set(report['results']) == {'1', '2'}
    assert report['metadata']['checkpoint']['sha256'] == _sha256_file(checkpoint)
    tokenizer_info = report['metadata']['tokenizer']
    assert tokenizer_info['sha256'] == _sha256_file(tokenizer_info['snapshot'])
    assert report['metadata']['datasets']['ctm_fixture']['splits']['test']['fingerprint']
    for evaluated in report['evaluations'].values():
        assert len(evaluated['samples']['ctm_fixture']) == 2
        assert evaluated['configs']['ctm_fixture']['dataset_path'] == 'json'
        assert evaluated['versions']['ctm_fixture'] == 1

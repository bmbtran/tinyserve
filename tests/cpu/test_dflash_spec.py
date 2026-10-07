"""M7 on CPU (float64): DFlash draft parity, lossless spec decoding, reference-loop
parity, and the spec x preemption x prefix-cache interactions."""

import math
import random

import pytest
import torch

from tests.reference import dflash_ref
from tests.util import make_tiny_model, tiny_engine, tiny_qwen3_config
from tinyserve.sampling import SamplingParams
from tinyserve.sequence import Sequence
from tinyserve.spec.proposer import DFlashProposer, OracleProposer, RandomProposer

VOCAB, MASK, BLK, BS = 512, 511, 16, 16
TARGET_IDS = [0, 2]
N_TOK = 64


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    """(target_path, hf_target, draft_path, ref_draft): 4-layer target, 2-layer random DFlash draft."""
    tpath = tmp_path_factory.mktemp("tiny_target4")
    hf_target = make_tiny_model(tpath, seed=1, num_hidden_layers=4)
    dcfg = tiny_qwen3_config(num_hidden_layers=2, block_size=BLK, num_target_layers=4,
                             dflash_config={"mask_token_id": MASK, "target_layer_ids": TARGET_IDS})
    dcfg._attn_implementation = "sdpa"
    torch.manual_seed(2)
    ref_draft = dflash_ref.DFlashDraftModel(dcfg).to(torch.float64).eval()
    dpath = tmp_path_factory.mktemp("tiny_dflash")
    ref_draft.save_pretrained(dpath, safe_serialization=True)
    return tpath, hf_target, dpath, ref_draft


@pytest.fixture(scope="module")
def prompts():
    rng = random.Random(5)
    return [[rng.randrange(MASK) for _ in range(rng.randrange(5, 80))] for _ in range(16)]


@pytest.fixture(scope="module")
def greedy(models, prompts):
    """Non-spec greedy continuations (N_TOK + 2*BLK long, so the oracle never runs out)."""
    tpath = models[0]
    outs = tiny_engine(tpath).generate(prompts, SamplingParams(max_tokens=N_TOK + 2 * BLK, ignore_eos=True))
    return [o["token_ids"] for o in outs]


def dflash_engine(models, **kw):
    tpath, _, dpath, _ = models
    return tiny_engine(tpath, spec_method="dflash", spec_draft_model=str(dpath), **kw)


@torch.no_grad()
def ref_draft_logits(models, tokens: list[int]) -> torch.Tensor:
    """Reference draft logits [15, V] for a sequence whose committed tokens are
    tokens[:-1] and whose last token is tokens[-1], computed from scratch (no cache)."""
    _, hf, _, ref = models
    n = len(tokens)
    hs = hf(torch.tensor([tokens[:-1]]), output_hidden_states=True).hidden_states
    ctx = dflash_ref.extract_context_feature(hs, TARGET_IDS)
    noise = hf.model.embed_tokens(torch.tensor([[tokens[-1]] + [MASK] * (BLK - 1)]))
    out = ref(target_hidden=ctx, noise_embedding=noise, position_ids=torch.arange(n - 1 + BLK)[None],
              past_key_values=None, use_cache=False, is_causal=False)
    return hf.lm_head(out[:, 1:])[0]


# ---------------------------------------------------------------------------
def test_1_draft_forward_parity(models):
    eng = dflash_engine(models)
    prop = eng.runner.proposer
    prop.record_logits = True
    _, hf, _, _ = models
    for L in (1, 7, 33):
        tokens = list(range(3, 3 + L)) + [42]
        seq = Sequence(tokens, SamplingParams())
        seq.block_table = list(range(math.ceil((len(tokens) + BLK) / BS)))
        with torch.no_grad():
            hs = hf(torch.tensor([tokens[:-1]]), output_hidden_states=True).hidden_states
        seq.pending_hidden = dflash_ref.extract_context_feature(hs, TARGET_IDS)[0]
        seq.draft_ctx_len = 0
        prop.propose([seq], eng.runner)
        ours = prop.logits_log[seq.seq_id][0]
        diff = (ours - ref_draft_logits(models, tokens)).abs().max().item()
        assert diff < 1e-9, (L, diff)
        assert seq.draft_ctx_len == len(tokens) - 1 and seq.pending_hidden is None


@pytest.mark.parametrize("method", ["oracle", "random", "dflash"])
def test_2_lossless(models, prompts, greedy, method):
    tpath = models[0]
    if method == "dflash":
        eng = dflash_engine(models, max_num_seqs=8)
    else:
        prop = OracleProposer({tuple(p): g for p, g in zip(prompts, greedy)}) if method == "oracle" else RandomProposer(VOCAB, seed=3)
        eng = tiny_engine(tpath, spec_method=method, max_num_seqs=8, proposer=prop)
    outs = eng.generate(prompts, SamplingParams(max_tokens=N_TOK, ignore_eos=True))
    assert [o["token_ids"] for o in outs] == [g[:N_TOK] for g in greedy]
    assert eng.metrics()["steps"]["SPEC"] > 0
    taus = [o["tau"] for o in outs]
    if method == "oracle":
        assert all(h == BLK for o in outs for h in o["spec_history"])
        assert taus == [16.0] * len(outs)
    if method == "random":
        assert max(taus) <= 1.1
    assert eng.block_manager.num_used_blocks == 0


def test_3_reference_loop_parity(models, prompts):
    """bs=1: same acceptance-length sequence AND the same draft logits at every
    step as the vendored z-lab spec_generate."""
    _, hf, _, ref = models
    eng = dflash_engine(models, max_num_seqs=1)
    eng.runner.proposer.record_logits = True
    for p in prompts[:4]:
        max_new = 48
        out_ids, acc, ref_logits = ref.spec_generate(target=hf, input_ids=torch.tensor([p]), max_new_tokens=max_new,
                                                     stop_token_ids=None, temperature=0.0, return_stats=True)
        # The reference loops while sum(acc) < max_new; we stop at 1 + sum(acc) >= max_tokens.
        seq = eng.add_request(p, SamplingParams(max_tokens=max_new + 1, ignore_eos=True))
        while eng.has_work():
            eng.step()
        assert seq.spec_history == acc
        assert seq.completion_token_ids[:max_new] == out_ids[0, len(p): len(p) + max_new].tolist()
        ours = eng.runner.proposer.logits_log[seq.seq_id]
        assert len(ours) == len(ref_logits)
        assert max((a - b).abs().max().item() for a, b in zip(ours, ref_logits)) < 1e-9


class CorruptedOracleDFlash(DFlashProposer):
    """Runs the REAL DFlash draft forward (all KV / position bookkeeping, logits
    logged) but proposes the target's greedy tokens with one token corrupted at
    a step-dependent index, so acceptance lengths vary over 1..16."""

    def __init__(self, draft, continuations):
        super().__init__(draft)
        self.cont = continuations
        self.record_logits = True
        self.step_tokens: dict[int, list[list[int]]] = {}

    def propose(self, seqs, runner):
        snapshot = {s.seq_id: list(s.token_ids) for s in seqs}
        super().propose(seqs, runner)
        rows = []
        for s in seqs:
            self.step_tokens.setdefault(s.seq_id, []).append(snapshot[s.seq_id])
            c = s.num_completion_tokens
            row = list(self.cont[tuple(s.token_ids[: s.num_prompt_tokens])][c : c + BLK - 1])
            j = (len(self.step_tokens[s.seq_id]) * 7 + s.seq_id) % BLK
            if j < BLK - 1:
                row[j] = (row[j] + 1) % MASK
            rows.append(row)
        return torch.tensor(rows)


def test_3b_draft_bookkeeping_with_variable_acceptance(models, prompts, greedy):
    tpath, _, dpath, _ = models
    eng = dflash_engine(models, max_num_seqs=4)
    prop = CorruptedOracleDFlash(eng.runner.draft_model, {tuple(p): g for p, g in zip(prompts, greedy)})
    eng.runner.proposer = prop
    outs = eng.generate(prompts[:4], SamplingParams(max_tokens=N_TOK, ignore_eos=True))
    assert [o["token_ids"] for o in outs] == [g[:N_TOK] for g in greedy[:4]]
    hist = [h for o in outs for h in o["spec_history"]]
    assert len(set(hist)) >= 5, hist  # many different acceptance lengths exercised
    checked = 0
    for sid, steps in prop.step_tokens.items():
        for tokens, logits in zip(steps, prop.logits_log[sid]):
            assert (logits - ref_draft_logits(models, tokens)).abs().max().item() < 1e-9
            checked += 1
    assert checked >= 20


@pytest.mark.parametrize("method", ["dflash", "oracle"])
def test_4_spec_with_preemption(models, prompts, greedy, method):
    tpath = models[0]
    kw = dict(max_num_seqs=8, num_kv_blocks=14, block_size=BS)
    if method == "dflash":
        eng = dflash_engine(models, **kw)
    else:
        eng = tiny_engine(tpath, spec_method="oracle", proposer=OracleProposer({tuple(p): g for p, g in zip(prompts, greedy)}), **kw)
    outs = eng.generate(prompts, SamplingParams(max_tokens=N_TOK, ignore_eos=True))
    assert eng.scheduler.num_preemptions > 0
    assert [o["token_ids"] for o in outs] == [g[:N_TOK] for g in greedy]


def test_5_prefix_cache_with_spec(models, greedy, prompts):
    """Cached blocks must carry valid DRAFT KV: draft logits with the cache on
    equal those with it off, and outputs stay lossless."""
    rng = random.Random(9)
    prefix = [rng.randrange(MASK) for _ in range(70)]
    reqs = [prefix + [rng.randrange(MASK) for _ in range(5)] for _ in range(4)]
    runs = {}
    for cache in (False, True):
        eng = dflash_engine(models, enable_prefix_cache=cache, max_num_seqs=4)
        eng.runner.proposer.record_logits = True
        first = eng.generate(reqs[:1], SamplingParams(max_tokens=40, ignore_eos=True))
        rest = eng.generate(reqs[1:], SamplingParams(max_tokens=40, ignore_eos=True))
        runs[cache] = (eng, first + rest)
    eng_on, on = runs[True]
    _, off = runs[False]
    assert [o["token_ids"] for o in on] == [o["token_ids"] for o in off]
    assert eng_on.metrics()["prefix_hit_tokens"] == 3 * 64  # 4 full blocks of the 70-token prefix
    ref = tiny_engine(models[0]).generate(reqs, SamplingParams(max_tokens=40, ignore_eos=True))
    assert [o["token_ids"] for o in on] == [o["token_ids"] for o in ref]
    log_on = list(runs[True][0].runner.proposer.logits_log.values())
    log_off = list(runs[False][0].runner.proposer.logits_log.values())
    for a, b in zip(log_on, log_off):
        assert len(a) == len(b)
        assert max((x - y).abs().max().item() for x, y in zip(a, b)) < 1e-9


def test_5b_commit_waits_for_draft_kv(models):
    """After PREFILL (before any spec step) the draft has not seen the prompt,
    so no new block may be registered in the prefix cache."""
    eng = dflash_engine(models)
    seq = eng.add_request(list(range(1, 50)), SamplingParams(max_tokens=30, ignore_eos=True))
    batch, _ = eng.step()
    assert batch.kind.name == "PREFILL"
    assert seq.draft_ctx_len == 0 and seq.pending_hidden.shape[0] == 49
    assert all(eng.block_manager.blocks[b].hash is None for b in seq.block_table)
    batch, _ = eng.step()
    assert batch.kind.name == "SPEC"
    committed = sum(eng.block_manager.blocks[b].hash is not None for b in seq.block_table)
    assert committed == min(seq.num_tokens - 1, seq.draft_ctx_len) // BS == 3

# speculative-decoding-demo

Speculative decoding built from scratch: a small draft model proposes tokens, a
large target model verifies them in a single forward pass, and the emitted text
is provably distributed exactly as the target model's own sampling. Includes a
latency benchmark against standard autoregressive decoding and an original
adaptive-k controller that tunes the proposal length on the fly.

## Problem

Autoregressive decoding is slow because each new token needs its own forward
pass through a large model, and those passes are serial. Speculative decoding
attacks the latency without changing the output: a cheap draft model guesses the
next several tokens, and the expensive target model checks all of them at once.
The subtlety is correctness. A naive "trust the draft" scheme would shift the
output distribution. The accept/reject rule used here keeps the output
statistically identical to sampling from the target directly, so you get speed
for free whenever the draft happens to agree with the target.

## Approach

- Draft proposes k tokens autoregressively using its own logits, and we record
  the draft probability it assigned to each proposed token.
- Target scores the prompt plus all k proposals in one forward pass, giving a
  target distribution at every position.
- Each proposed token is accepted with probability `min(1, p_target/p_draft)`.
  On the first rejection we resample that position from the normalized residual
  `max(0, p_target - p_draft)` and stop the round. If all k are accepted we also
  sample one bonus token from the target at the final position.
- Adaptive-k (the original addition): an EMA of the observed acceptance rate
  drives k up when the draft is agreeing often and down when it is not, bounded
  by `[k_min, k_max]`, so the proposal length self-tunes per prompt.
- A benchmark harness times standard vs fixed-k vs adaptive-k under matched
  seeds and reports tokens/sec, wall time, speedup, and mean acceptance rate.

### Why the accept rule is exact

This is rejection sampling. Let `q = p_draft` and `p = p_target` at a position.
Draw a candidate `x ~ q` and accept it with probability `min(1, p(x)/q(x))`. If
rejected, draw a replacement from the normalized residual
`(p - q)_+ / sum((p - q)_+)`. The mixture of "accepted candidate" and
"residual resample" reproduces `p` exactly. So every emitted token is a genuine
sample from the target; the draft only decides how many target forward passes we
can skip. High agreement means many tokens per target pass, which is where the
speedup comes from.

### KV-cache note

For clarity this implementation recomputes the forward pass over the growing
sequence each round instead of threading manual `past_key_values` through both
models. The accept/reject bookkeeping stays readable and the output distribution
is unchanged. The measured speedup comes from the target running once per round
over `k+1` positions rather than once per token; adding KV caching would lower
the constant factor further but not change the algorithm. This trade-off is
documented at the top of `src/spec_decode.py`.

## Setup

```bash
# Create and activate a virtual environment (Python 3.12).
uv venv --python 3.12 .venv        # or: python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

# Install torch from the CUDA 12.8 wheel index (RTX 5090 / sm_120), then the rest.
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Optional HF token (public GPT-2 models do not require one).
cp .env.example .env
```

Models default to draft `distilgpt2` and target `gpt2-large`. If `gpt2-large` is
too heavy for your machine, pass `--target gpt2-medium` to either script.

## How to run

Generate text with speculative decoding:

```bash
python scripts/run_generate.py \
  --prompt "The history of computing began" \
  --max-new-tokens 128 --k 4
```

Use the adaptive-k controller instead of a fixed k:

```bash
python scripts/run_generate.py \
  --prompt "The history of computing began" \
  --max-new-tokens 128 --adaptive --k-min 1 --k-max 10
```

Run the benchmark and save the report to `outputs/bench.json`:

```bash
python scripts/run_bench.py \
  --prompt "In a distant future, humanity" \
  --max-new-tokens 128 --k 4 --k-min 1 --k-max 10
```

Run the tests (no large download; model-loading test auto-skips when offline):

```bash
pytest -q
```

## Results

Run the commands above to produce the numbers. Numbers below are produced by
running the commands above; this repo ships the code, run it to populate them.

Expected qualitative behavior:

- Speculative decoding matches the target model's output distribution. With
  draft equal to target the acceptance probability is 1 and the output is
  identical to plain target sampling (this is asserted in the tests).
- Wall time drops relative to the standard baseline whenever the draft agrees
  with the target often. The speedup grows with the acceptance rate, since a
  higher acceptance rate means more tokens emitted per target forward pass.
- When acceptance is low, fixed large k wastes work; adaptive k should shrink
  the proposal length automatically and recover, while high-acceptance prompts
  push it toward k_max. Adaptive k should land near a good proposal length
  without manual tuning.
- On CPU or with a weak draft the speculative path can be slower than the
  baseline (the draft and the wider target pass cost more than they save); the
  benefit shows up on GPU with a fast draft and a large target.

| method                     | wall (s) | tok/s  | speedup | acc. rate | mean k |
| -------------------------- | -------- | ------ | ------- | --------- | ------ |
| standard                   | TBD (run)| TBD    | 1.00x   | -         | -      |
| speculative_fixed_k=4      | TBD (run)| TBD    | TBD     | TBD       | 4.00   |
| speculative_adaptive_k     | TBD (run)| TBD    | TBD     | TBD       | TBD    |

## What I'd do next at larger scale

Thread real KV caches through both models so each round only runs the target
over the new `k+1` positions instead of the whole sequence, and batch multiple
draft continuations (a tree of proposals) so one target pass verifies several
candidate paths. At that point the controller would tune both proposal length
and tree width against measured per-token latency rather than acceptance rate
alone.

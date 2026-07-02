# Corrections log

## 2026-07-02 — Corrected rescoring baseline results

The first integrated rescoring baseline used plain beam search for Whisper N-best generation.
That produced degenerate 10-best lists (all 10 candidates collapsed to the same text), which made
`oracle` and `rescoring` appear identical. A later diversity audit in the working project showed
that this was an artifact of the N-best generation method, not the rescoring logic itself.

The public-release rescoring materials were corrected to use the verified sampled N-best pool
from `rescoring_experiment_v3/` with the following defaults:

- `do_sample=True`
- `temperature=0.7`
- `top_p=0.9`
- `num_beams=1`
- `num_return_sequences=10`
- `seed=42`

The corrected release outputs now report the v3 values in `experiments/rescoring/summaries/`,
`RESULTS.md`, `FINDINGS.md`, and the rescoring documentation.

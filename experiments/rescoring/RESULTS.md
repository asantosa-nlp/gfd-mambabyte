# Rescoring Baseline Results

- JV N-best file: `experiments/rescoring/nbest/jv_t07_topp09_s42/jv_nbest.jsonl`
- SU N-best file: `experiments/rescoring/nbest/su_t07_topp09_s42/su_nbest.jsonl`
- JV permutation-test p-value: `0.000100` (seed 42)
- SU permutation-test p-value: `0.000100` (seed 42)

| System | JV WER | SU WER | JV CER | SU CER |
|---|---:|---:|---:|---:|
| Zero-shot Whisper (1-best) | 58.16 | 61.64 | 14.44 | 12.70 |
| N-best rescoring (adapted LM, post-hoc, same λ/α) | 60.53 | 63.21 | 15.47 | 13.30 |
| N-best oracle (lowest-WER in 10-best) | 47.72 | 52.29 | 11.75 | 11.39 |
| GFD first-pass fusion (our method) | 51.87 | 54.72 | 15.62 | 13.33 |

## Per-language deltas
- JV: rescoring vs GFD = +8.66 pp; oracle gap = -12.80 pp
- SU: rescoring vs GFD = +8.49 pp; oracle gap = -10.92 pp

## Permutation test
- JV GFD-vs-rescoring mean delta: `-0.086580`
- JV p-value: `0.000100`
- SU GFD-vs-rescoring mean delta: `-0.084892`
- SU p-value: `0.000100`

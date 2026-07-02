# Rescoring correction report

Date: 2026-07-02

## Source of truth used

Corrected v3 outputs were verified in the working project snapshot `rescoring_experiment_v3/`.

Verified values from the v3 summaries:

- **Javanese**: zero-shot 58.16 / 14.44, rescoring 60.53 / 15.47, oracle 47.72 / 11.75, GFD 51.87 / 15.62, p = 0.0001
- **Sundanese**: zero-shot 61.64 / 12.70, rescoring 63.21 / 13.30, oracle 52.29 / 11.39, GFD 54.72 / 13.33, p = 0.0001

## Files changed

1. `experiments/rescoring/summaries/jv_summary.json` — replaced the broken v1 summary values with the verified v3 numbers and repo-relative N-best path.
2. `experiments/rescoring/summaries/su_summary.json` — replaced the broken v1 summary values with the verified v3 numbers and repo-relative N-best path.
3. `experiments/rescoring/RESULTS.md` — rewrote the table and supporting bullets to reflect the corrected v3 results.
4. `experiments/rescoring/FINDINGS.md` — rewrote the interpretation to state the corrected relationship between rescoring, oracle, zero-shot, and GFD.
5. `experiments/rescoring/README.md` — updated the experiment README to describe the corrected v3 result, sampling-based N-best generation, and the new fixed N-best path names.
6. `experiments/rescoring/build_rescoring_report.py` — updated the report-generation text so rerunning the builder recreates the corrected findings.
7. `experiments/rescoring/generate_whisper_nbest.py` — changed the N-best generation defaults to the verified sampling configuration.
8. `experiments/rescoring/run_whisper_nbest.sh` — updated the helper defaults and forwarded the sampling parameters to the Python generator.
9. `experiments/rescoring/run_rescore_whisper_nbest.sh` — updated the default N-best lookup to prefer the corrected v3 path names.
10. `configs/rescoring.yaml` — added the corrected N-best generation defaults and updated the canonical repo-relative N-best paths.
11. `README.md` — updated the top-level experiment summary to say the oracle remains better than GFD.
12. `docs/REPRODUCE.md` — updated the rescoring subsection to document the corrected sampling defaults and corrected v3 numbers.
13. `experiments/rescoring/CORRECTIONS.md` — added a transparency note describing the correction from broken beam-search N-best generation to sampled N-best generation.

## Before / after summary

### Broken v1 shipped result

- Javanese rescoring: **65.80% WER / 22.97% CER**
- Javanese oracle: **65.80% WER / 22.97% CER**
- Sundanese rescoring: **60.97% WER / 12.58% CER**
- Sundanese oracle: **60.97% WER / 12.58% CER**
- Narrative issue: the oracle equaled rescoring because the N-best pool was degenerate (beam collapse), not because the reranker was oracle-optimal.

### Corrected v3 result now shipped

- Javanese rescoring: **60.53% WER / 15.47% CER**
- Javanese oracle: **47.72% WER / 11.75% CER**
- Sundanese rescoring: **63.21% WER / 13.30% CER**
- Sundanese oracle: **52.29% WER / 11.39% CER**
- GFD remains unchanged at **51.87% / 15.62%** (JV) and **54.72% / 13.33%** (SU)
- GFD still beats post-hoc rescoring in both languages, but the true oracle beats GFD in both languages.

## Generation-default fix

The only code-path change affecting decoding behavior was the N-best generation default update in `experiments/rescoring/generate_whisper_nbest.py` (and the matching helper wrapper defaults).

### Before

- `do_sample = false`
- `num_beams = 10`
- `temperature = unset`
- `top_p = unset`
- `seed = unset`

### After

- `do_sample = true`
- `num_beams = 1`
- `temperature = 0.7`
- `top_p = 0.9`
- `num_return_sequences = 10`
- `seed = 42`

The rescoring helper now prefers the corrected fixed N-best paths:

- `experiments/rescoring/nbest/jv_t07_topp09_s42/jv_nbest.jsonl`
- `experiments/rescoring/nbest/su_t07_topp09_s42/su_nbest.jsonl`

## Safety / hygiene checks

- No absolute home-path string was introduced in the edited files.
- No secret, token, password, or credential string was introduced.
- The zero-shot / GFD baseline prediction files in `experiments/rescoring/baseline_predictions/` were not modified.
- No other repo area outside `experiments/rescoring/`, `configs/rescoring.yaml`, `README.md`, and `docs/REPRODUCE.md` was changed.
- The corrected summary JSONs are valid JSON and still use repo-relative N-best path strings only.

## Verification performed

- Verified the corrected v3 numbers against the source summaries in `rescoring_experiment_v3/`.
- Regenerated `experiments/rescoring/RESULTS.md` and `experiments/rescoring/FINDINGS.md` from the corrected summaries.
- Ran syntax checks on the edited Python and shell scripts.
- Searched the edited files for stale oracle=rescoring / beam-collapse wording and for absolute home paths; none remain.

## Remaining note

The raw Whisper N-best JSONL files remain gitignored, as intended. The release repo now documents the corrected sampling defaults and the corrected v3 result transparently.

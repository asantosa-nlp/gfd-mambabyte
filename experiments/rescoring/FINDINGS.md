# Findings

This report summarizes the corrected post-hoc N-best rescoring baseline against the first-pass GFD method.

## Javanese
- Zero-shot Whisper WER: 58.16%
- Post-hoc rescoring WER: 60.53%
- 10-best oracle WER: 47.72%
- GFD WER: 51.87%
- GFD significantly outperforms post-hoc rescoring (paired permutation p = `0.000100`, seed 42, 10,000 permutations).
- Post-hoc rescoring is worse than zero-shot by +2.37 pp.
- The true oracle outperforms GFD by 4.14 pp, so the reranker does not always select the reference-optimal hypothesis.

## Sundanese
- Zero-shot Whisper WER: 61.64%
- Post-hoc rescoring WER: 63.21%
- 10-best oracle WER: 52.29%
- GFD WER: 54.72%
- GFD significantly outperforms post-hoc rescoring (paired permutation p = `0.000100`, seed 42, 10,000 permutations).
- Post-hoc rescoring is worse than zero-shot by +1.57 pp.
- The true oracle outperforms GFD by 2.43 pp, so the reranker does not always select the reference-optimal hypothesis.

## Interpretation
The corrected N-best pool is genuinely diverse, but the post-hoc rescoring baseline remains weaker than GFD and also worse than zero-shot in both languages. The oracle is stronger than GFD, so the earlier oracle=rescoring equality was a beam-collapse artifact, not the true result.

## Next step
If you want the exact manuscript-ready wording, cite `RESULTS.md` and note that the N-best candidates were regenerated with sampling-based generation (temperature 0.7, top_p 0.9, num_beams 1, seed 42).
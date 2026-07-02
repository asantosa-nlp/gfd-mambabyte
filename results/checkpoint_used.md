# Checkpoint used for the reported Table 4 rows

The Table 4 GFD + MambaByte rows use the **20k checkpoint**:
`checkpoints/mb_cpt_{jv,su}_v5_clean/step_0020000.pt`.

- Javanese decode config: `configs/decode_javanese.yaml`
- Sundanese decode config: `configs/decode_sundanese.yaml`

The checkpoint-selection sweep evaluates 10k / 20k / 30k / 40k on the 1,000-utterance search set and selects **20k**.

Zero-shot Whisper rows and the English-base ablation rows use **no CPT checkpoint**.

The post-hoc rescoring baseline in `experiments/rescoring/` also uses the same
step-20k adapted checkpoints; it differs only in applying the LM after Whisper
produces a fixed N-best list.

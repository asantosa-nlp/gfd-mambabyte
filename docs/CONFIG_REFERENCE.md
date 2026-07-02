# Configuration reference

This file maps paper hyperparameters to the release configs.

## CPT (LoRA continual pre-training)

| Paper item | Config key | Javanese file | Sundanese file |
|---|---|---|---|
| Base LM | `lm.model_name_or_path` | `configs/cpt_javanese.yaml` | `configs/cpt_sundanese.yaml` |
| LoRA rank | `lm.lora_config.rank` | `32` | `32` |
| LoRA alpha | `lm.lora_config.alpha` | `64` | `64` |
| LoRA dropout | `lm.lora_config.dropout` | `0.05` | `0.05` |
| LR schedule | `train.lr`, `train.min_lr`, `train.warmup_steps` | `2e-4 → 1e-5`, `1000` warm-up | same |
| Batch / grad-accum | `train.batch_size`, `train.grad_accum_steps` | `4 / 4` | same |
| Precision | `lm.device` + `cpt/train.py` dtype | float32 | float32 |
| Total steps | `train.total_steps` | `40000` | `40000` |
| Checkpoint cadence | `train.save_every` | `1000` | `1000` |
| Training corpus | `train.text_file` | `data/text_corpus/combined_jv_v2_clean.txt` | `data/text_corpus/combined_su_v2_clean.txt` |

## Decode / GFD operating points

| Paper item | Config key | Javanese file | Sundanese file |
|---|---|---|---|
| Fusion weight λ | `fusion.lambda` | `0.70` | `0.80` |
| LM weight α | `fusion.lm_logit_alpha` | `0.20` | `0.20` |
| Beam width W | `fusion.beam_width` | `5` | `5` |
| Top-K K | `fusion.top_k_candidates` | `20` | `20` |
| Min output bytes | `fusion.min_output_bytes` | `30` | `15` |
| Max output bytes | `fusion.max_output_bytes` | `160` | `160` |
| Trie backend | `fusion.decoder_backend` | `cached_trie` | `cached_trie` |
| LM precision | `lm.dtype` | `float32` | `float32` |
| Whisper precision | `asr.dtype` | `float16` | `float16` |
| Whisper language | `whisper_lang_code` | `jw` | `su` |
| Test manifest | `eval.blind_manifest` | `data/speech_corpus/local_jv/test.jsonl` | `data/speech_corpus/local_su/test.jsonl` |

## Prefix-trie state precision

The cached MambaByte trie stores the recurrent state on CPU as the same dtype as the loaded LM.
In the release configs the LM is `float32`, so trie states are also `float32`.

Let:

- `L = 48` layers
- `d_model = 1792`
- `expand = 2` → internal width `d_inner = 3584`
- `d_state = 16`
- conv cache length = 4 slots per layer in the implementation used here

Let `conv_cache_len = 4`. The per-node state size is the sum of the convolution cache and SSM cache for every layer:

`bytes/node = L × d_inner × (conv_cache_len + d_state) × 4`

Substituting the values above gives:

`48 × (3584 × 4 + 3584 × 16) × 4 = 13,762,560 bytes ≈ 13.13 MiB`

That is the origin of the paper's ~13 MB estimate.

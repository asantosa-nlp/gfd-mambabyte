#!/usr/bin/env python3
"""
Whisper-only baseline eval (no GFD, no LM).
Measures pure ASR WER/CER on the blind manifest to establish the
zero-shot Whisper baseline that GFD is compared against.

Usage:
    python -u scripts/run_whisper_baseline.py \
        --model openai/whisper-large-v3 \
        --lang-code jw \
        --manifest data/speech_corpus/openslr_jv/gfd_blind_filt_seed42.jsonl \
        --out-dir results/baseline_whisper_large_jv \
        --dtype float16
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import jiwer
import librosa
import torch


def normalize_text(text: str, lowercase: bool = True,
                   strip_punct: bool = True, form: str = "NFC") -> str:
    text = unicodedata.normalize(form, text)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def load_manifest(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="openai/whisper-large-v3")
    parser.add_argument("--processor-model", default=None,
                        help="Load processor from this model instead of --model. "
                             "Useful when fine-tuned dirs lack vocab.json (slow tokenizer). "
                             "Defaults to --model.")
    parser.add_argument("--lang-code", required=True,
                        help="Whisper language code, e.g. 'jw' for Javanese, 'su' for Sundanese")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--dtype",    default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--initial-prompt", default=None,
                        help="Optional Whisper initial_prompt for language steering")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16,
                 "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    os.makedirs(args.out_dir, exist_ok=True)
    records = load_manifest(args.manifest)
    if args.max_utterances:
        records = records[: args.max_utterances]

    print(f"[baseline] model   : {args.model}")
    print(f"[baseline] lang    : {args.lang_code}")
    print(f"[baseline] dtype   : {args.dtype}")
    print(f"[baseline] manifest: {args.manifest}  ({len(records)} utterances)")
    print(f"[baseline] out-dir : {args.out_dir}")
    if args.initial_prompt:
        print(f"[baseline] prompt  : {args.initial_prompt!r}")
    print()

    print("[baseline] Loading model ...")
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor_src = args.processor_model if args.processor_model else args.model
    processor = WhisperProcessor.from_pretrained(processor_src)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype
    ).to("cuda")
    model.eval()

    # Build forced decoder IDs: language + transcribe task
    forced_ids = processor.get_decoder_prompt_ids(
        language=args.lang_code, task="transcribe"
    )
    # Reserve headroom: forced_ids + BOS/SOT tokens can be ~4 tokens total.
    # max_new_tokens must satisfy: len(decoder_input_ids) + max_new_tokens <= max_target_positions
    _max_pos = model.config.max_target_positions  # 448 for large-v3
    _overhead = len(forced_ids) + 2               # forced pairs + start tokens
    _max_new_tokens = _max_pos - _overhead

    print("[baseline] Model loaded. Starting eval ...\n")
    print(f"{'='*70}")
    print(f"{'#':>5}  {'WER':>6}  {'CER':>6}  {'ms':>6}  reference / hypothesis")
    print(f"{'='*70}")

    predictions = []
    total_wer = total_cer = total_audio_s = total_decode_s = 0.0

    for i, rec in enumerate(records):
        audio_path = rec["path"]
        reference_raw = rec.get("sentence", "")
        duration = rec.get("duration", 0.0)

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        inputs = processor(audio, sampling_rate=16000,
                           return_tensors="pt").input_features.to("cuda", dtype=dtype)

        t0 = time.time()
        with torch.no_grad():
            gen_kwargs = dict(
                forced_decoder_ids=forced_ids,
                max_new_tokens=_max_new_tokens,
            )
            if args.initial_prompt:
                prompt_ids = processor.get_prompt_ids(args.initial_prompt)
                gen_kwargs["prompt_ids"] = torch.tensor(
                    prompt_ids, dtype=torch.long, device="cuda"
                ).unsqueeze(0)

            token_ids = model.generate(inputs, **gen_kwargs)

        decode_ms = (time.time() - t0) * 1000.0

        hypothesis_raw = processor.batch_decode(
            token_ids, skip_special_tokens=True
        )[0]

        reference = normalize_text(reference_raw)
        hypothesis = normalize_text(hypothesis_raw)

        wer = float(jiwer.wer(reference, hypothesis)) if reference else 0.0
        cer = float(jiwer.cer(reference, hypothesis)) if reference else 0.0

        total_wer   += wer
        total_cer   += cer
        total_audio_s  += duration
        total_decode_s += decode_ms / 1000.0

        uid = rec.get("speaker_id", str(i)) + f"_{i:04d}"
        predictions.append({
            "utterance_id":    uid,
            "audio_path":      audio_path,
            "reference_raw":   reference_raw,
            "hypothesis_raw":  hypothesis_raw,
            "reference":       reference,
            "hypothesis":      hypothesis,
            "wer":             round(wer, 4),
            "cer":             round(cer, 4),
            "duration_s":      duration,
            "decode_ms":       round(decode_ms, 1),
        })

        print(
            f"{i+1:>5}  {wer:>6.3f}  {cer:>6.3f}  {decode_ms:>6.0f}  "
            f"{reference[:40]!r} / {hypothesis[:40]!r}"
        )
        if (i + 1) % 10 == 0:
            sys.stdout.flush()

    n = len(predictions)
    mean_wer = total_wer / max(n, 1)
    mean_cer = total_cer / max(n, 1)
    rtf = total_decode_s / max(total_audio_s, 1e-9)

    pred_path = os.path.join(args.out_dir, "predictions.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    summary = {
        "model":           args.model,
        "lang_code":       args.lang_code,
        "initial_prompt":  args.initial_prompt,
        "manifest":        args.manifest,
        "n_utterances":    n,
        "mean_wer":        round(mean_wer, 4),
        "mean_cer":        round(mean_cer, 4),
        "rtf":             round(rtf, 4),
        "dtype":           args.dtype,
    }
    sum_path = os.path.join(args.out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[baseline] Results — {args.model}  lang={args.lang_code}")
    print(f"  Mean WER : {mean_wer:.4f}  ({mean_wer*100:.2f}%)")
    print(f"  Mean CER : {mean_cer:.4f}  ({mean_cer*100:.2f}%)")
    print(f"  RTF      : {rtf:.4f}")
    print(f"\n  predictions → {pred_path}")
    print(f"  summary     → {sum_path}")


if __name__ == "__main__":
    main()

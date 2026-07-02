#!/usr/bin/env python3
"""
GFD + MambaByte evaluation runner.

Usage:
    python -u scripts/run_exp_mb.py configs/mb_jv.yaml --split blind
    python -u scripts/run_exp_mb.py configs/mb_jv.yaml --split search
    python -u scripts/run_exp_mb.py configs/mb_su.yaml --split blind
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

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
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_text(text: str, cfg: dict) -> str:
    form = cfg.get("unicode_form", "NFC")
    text = unicodedata.normalize(form, text)
    if cfg.get("lowercase", True):
        text = text.lower()
    if cfg.get("strip_punctuation", True):
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_lm(cfg: dict):
    lm_cfg = cfg["lm"]
    kind = lm_cfg["kind"]
    device = lm_cfg.get("device", "cuda")
    _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = _dtype_map.get(lm_cfg.get("dtype", "bfloat16"), torch.float16)

    if kind == "mambabyte":
        from gfd.lm_adapters import MambaByteAdapter
        return MambaByteAdapter(
            model_name_or_path=lm_cfg.get("model_name_or_path", "JunxiongWang/MambaByte_PG19_972M"),
            lora_ckpt=lm_cfg.get("lora_ckpt", None),
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unknown lm.kind: {kind!r}. Expected 'mambabyte'.")


def build_decoder_cfg(cfg: dict) -> dict:
    return {
        "lm": {"kind": cfg["lm"]["kind"]},
        "fusion": cfg["fusion"],
    }


def compute_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    wer = jiwer.wer(reference, hypothesis)
    cer = jiwer.cer(reference, hypothesis)
    return float(wer), float(cer)


def main():
    parser = argparse.ArgumentParser(description="GFD + MambaByte evaluation")
    parser.add_argument("config")
    parser.add_argument("--split", choices=["blind", "search"], default="blind")
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--trace", type=int, default=0, metavar="LEVEL")
    parser.add_argument("--progress", type=int, default=25, metavar="N",
                        help="print heartbeat every N decode steps (0=off)")
    parser.add_argument("--blind-manifest", default=None,
                        help="override eval.blind_manifest from config")
    parser.add_argument("--out-dir", default=None,
                        help="override output directory from config")
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"],
                        help="override dtype for both ASR and LM (e.g. float16 for V100)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from existing predictions.json; checkpoint every 50 utterances")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.trace:
        cfg.setdefault("fusion", {})["trace"] = args.trace
    cfg.setdefault("fusion", {})["progress_every"] = args.progress
    if args.dtype:
        cfg.setdefault("asr", {})["dtype"] = args.dtype
        cfg.setdefault("lm", {})["dtype"] = args.dtype
    exp_id = cfg.get("experiment_id", "gfd_mambabyte")
    norm_cfg = cfg["eval"].get("text_normalization", {})

    if args.split == "blind":
        manifest_path = args.blind_manifest or cfg["eval"]["blind_manifest"]
        out_dir = args.out_dir or cfg["output"]["blind_dir"]
    else:
        manifest_path = cfg["eval"]["search_manifest"]
        out_dir = args.out_dir or cfg["output"]["search_dir"]

    os.makedirs(out_dir, exist_ok=True)
    records = load_manifest(manifest_path)
    if args.max_utterances:
        records = records[: args.max_utterances]

    pred_path = os.path.join(out_dir, "predictions.json")

    # Resume: load existing predictions and skip completed utterances
    predictions = []
    total_wer = total_cer = total_audio_s = total_decode_s = 0.0
    skip_n = 0
    if args.resume and os.path.exists(pred_path):
        with open(pred_path) as f:
            predictions = json.load(f)
        skip_n = len(predictions)
        if skip_n >= len(records):
            print(f"[run_exp_mb] Already complete ({skip_n}/{len(records)} utterances). Nothing to do.")
            sys.exit(0)
        for p in predictions:
            total_wer += p["wer"]
            total_cer += p["cer"]
            total_audio_s += p.get("duration_s", 0.0)
            total_decode_s += p.get("decoder_stats", {}).get("decode_wall_ms", 0.0) / 1000.0
        records = records[skip_n:]
        print(f"[run_exp_mb] Resuming from utterance {skip_n + 1}/{skip_n + len(records)}")

    print(f"[run_exp_mb] {exp_id}  split={args.split}  utterances={skip_n + len(records)}")
    print(f"[run_exp_mb] manifest: {manifest_path}")
    print(f"[run_exp_mb] output:   {out_dir}")

    lora_ckpt = cfg["lm"].get("lora_ckpt")
    if lora_ckpt and not os.path.exists(lora_ckpt):
        print(f"\n[run_exp_mb] ERROR: LoRA checkpoint not found: {lora_ckpt}")
        print("[run_exp_mb] CPT training must complete before running evaluation.")
        sys.exit(1)

    print("\n[run_exp_mb] Loading ASR model ...")
    from gfd.asr_adapters import WhisperAdapter
    asr = WhisperAdapter(
        model_name_or_path=cfg["asr"]["model_name_or_path"],
        language_code=cfg["whisper_lang_code"],
        device=cfg["asr"].get("device", "cuda"),
        dtype=(torch.bfloat16 if cfg["asr"].get("dtype", "bfloat16") == "bfloat16"
               else torch.float16),
        initial_prompt=cfg["asr"].get("initial_prompt", None),
    )

    print("\n[run_exp_mb] Loading MambaByte LM ...")
    lm = build_lm(cfg)

    from gfd.byte_gfd import ByteLevelFusionDecoder
    decoder = ByteLevelFusionDecoder(asr=asr, lm=lm, cfg=build_decoder_cfg(cfg))

    print(f"\n{'='*70}")
    print(f"{'#':>5}  {'WER':>6}  {'CER':>6}  {'ms':>6}  reference / hypothesis")
    print(f"{'='*70}")

    for i, rec in enumerate(records):
        abs_i = skip_n + i
        audio_path = rec["path"]
        reference_raw = rec["sentence"]
        duration = rec.get("duration", 0.0)

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        uid = rec.get("speaker_id", str(i)) + f"_{i:04d}"
        result = decoder.decode(audio, utterance_id=uid)
        decode_ms = result["decoder_stats"]["decode_wall_ms"]

        hypothesis_raw = result["text"]
        reference = normalize_text(reference_raw, norm_cfg)
        hypothesis = normalize_text(hypothesis_raw, norm_cfg)

        wer, cer = compute_wer_cer(reference, hypothesis)
        total_wer += wer
        total_cer += cer
        total_audio_s += duration
        total_decode_s += decode_ms / 1000.0

        entry = {
            "utterance_id": rec.get("speaker_id", str(i)) + f"_{i:04d}",
            "audio_path": audio_path,
            "reference_raw": reference_raw,
            "hypothesis_raw": hypothesis_raw,
            "reference": reference,
            "hypothesis": hypothesis,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "duration_s": duration,
            "asr_log_prob": round(result["asr_log_prob"], 4),
            "lm_log_prob": round(result["lm_log_prob"], 4),
            "lm_forwards": result["lm_forwards"],
            "num_bytes": result["num_bytes"],
            "decoder_stats": result["decoder_stats"],
        }
        predictions.append(entry)

        print(
            f"{abs_i+1:>5}  {wer:>6.3f}  {cer:>6.3f}  {decode_ms:>6.0f}  "
            f"{reference[:40]!r} / {hypothesis[:40]!r}"
        )
        if (abs_i + 1) % 10 == 0:
            sys.stdout.flush()

        # Periodic checkpoint so the run is resumable if interrupted
        if args.resume and (abs_i + 1) % 50 == 0:
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    n = len(predictions)
    mean_wer = total_wer / max(n, 1)
    mean_cer = total_cer / max(n, 1)
    rtf = total_decode_s / max(total_audio_s, 1e-9)
    gate_stats = decoder.gate.stats()
    mean_lm_forwards = sum(p["lm_forwards"] for p in predictions) / max(n, 1)

    summary = {
        "experiment_id": exp_id,
        "split": args.split,
        "manifest": manifest_path,
        "n_utterances": n,
        "mean_wer": round(mean_wer, 4),
        "mean_cer": round(mean_cer, 4),
        "rtf": round(rtf, 4),
        "mean_lm_forwards": round(mean_lm_forwards, 2),
        "gate_stats": gate_stats,
        "config": cfg,
    }
    sum_path = os.path.join(out_dir, "summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[run_exp_mb] Results ({exp_id}, {args.split})")
    print(f"  Mean WER : {mean_wer:.4f}  ({mean_wer*100:.2f}%)")
    print(f"  Mean CER : {mean_cer:.4f}  ({mean_cer*100:.2f}%)")
    print(f"  RTF      : {rtf:.4f}")
    print(f"  LM calls : {mean_lm_forwards:.1f} / utterance")
    print(f"  Gate     : {gate_stats['invocations']} invoke / {gate_stats['skips']} skip")
    total_pen = sum(p["decoder_stats"].get("heuristic_penalties", 0) for p in predictions)
    print(f"  Heuristic penalties: {int(total_pen)} beams across {n} utterances")
    print(f"\n  predictions → {pred_path}")
    print(f"  summary     → {sum_path}")


if __name__ == "__main__":
    main()

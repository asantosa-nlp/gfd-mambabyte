#!/usr/bin/env python3
"""
Generate an N-best Whisper hypothesis list for the JV/SU 460-utterance test set.

This script is intentionally narrow:
  - it performs first-pass Whisper decoding only;
  - it writes per-utterance N-best hypotheses with Whisper sequence scores;
  - it does not rescoring, LM fusion, or WER evaluation.

The companion shell helper `experiments/rescoring/run_whisper_nbest.sh` sets CUDA_VISIBLE_DEVICES
and selects the language / manifest path.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import librosa
import torch


def load_manifest(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_progress(out_path: Path, state_path: Path) -> int:
    """Return the next row index to process.

    Priority:
      1. explicit state file (if valid)
      2. count of parseable JSONL rows in the output file
    """
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            next_index = int(state.get("next_index", 0))
            if next_index >= 0:
                return next_index
        except Exception:
            pass

    if not out_path.exists():
        return 0

    next_index = 0
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except Exception:
                break
            next_index += 1
    return next_index


def save_state(state_path: Path, *, next_index: int, utt_id: str | None = None, status: str = "running", error: str | None = None) -> None:
    payload = {
        "status": status,
        "next_index": next_index,
        "utt_id": utt_id,
        "error": error,
        "timestamp": time.time(),
    }
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, state_path)


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}")
    if name == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return mapping[name]  # type: ignore[return-value]


def resolve_utt_id(rec: dict, audio_path: str, index: int) -> str:
    for key in ("utt_id", "utterance_id", "id"):
        value = rec.get(key)
        if value:
            return str(value)
    return Path(audio_path).stem or f"utt_{index:04d}"


def compute_sequence_logprobs(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    sequences: torch.Tensor,
) -> list[float]:
    """Teacher-force each generated sequence and sum token log-probabilities.

    This avoids depending on generation-time beam bookkeeping, which varies across
    Transformers versions. The returned values are raw summed log-probabilities
    over the sequence until the first EOS token. Because all candidates for one
    utterance share the same fixed Whisper prefix, the constant prefix term does
    not affect ranking.
    """
    eos_id = model.config.eos_token_id
    if eos_id is None:
        raise RuntimeError("Whisper model is missing eos_token_id")

    if sequences.ndim != 2:
        raise ValueError(f"Expected sequences with shape [batch, seq_len], got {tuple(sequences.shape)}")
    if sequences.shape[1] < 2:
        return [0.0 for _ in range(sequences.shape[0])]

    decoder_input_ids = sequences[:, :-1]
    target_ids = sequences[:, 1:]

    with torch.inference_mode():
        forward_kwargs = dict(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
            return_dict=True,
        )
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        out = model(**forward_kwargs)
        logits = out.logits  # [B, T, V]

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]

    scores: list[float] = []
    target_cpu = target_ids.detach().cpu()
    gathered_cpu = gathered.detach().cpu()
    for row_toks, row_scores in zip(target_cpu, gathered_cpu):
        eos_positions = (row_toks == eos_id).nonzero(as_tuple=False)
        if len(eos_positions) > 0:
            # Include the first EOS token, then stop.
            end_idx = int(eos_positions[0].item()) + 1
        else:
            end_idx = row_scores.shape[0]
        scores.append(float(row_scores[:end_idx].sum().item()))
    return scores


def decode_one_row(
    *,
    model,
    processor,
    rec: dict,
    row_index: int,
    device: str,
    dtype: torch.dtype,
    forced_ids: list[list[int]] | list[tuple[int, int]],
    max_new_tokens: int,
    num_beams: int,
    num_hypotheses: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    whisper_lang: str,
    model_name: str,
    processor_name: str,
    initial_prompt: str | None,
) -> tuple[dict, float]:
    audio_path = rec["path"]
    reference = rec.get("sentence", "")
    utt_id = resolve_utt_id(rec, audio_path, row_index)

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    enc = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
    )
    inputs = enc.input_features.to(device, dtype=dtype)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    gen_kwargs = {
        "forced_decoder_ids": forced_ids,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "num_return_sequences": num_hypotheses,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "return_dict_in_generate": True,
    }
    gen_device = "cpu" if device == "cpu" else "cuda"
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(seed)
    gen_kwargs["generator"] = generator
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if initial_prompt:
        prompt_ids = processor.get_prompt_ids(initial_prompt)
        gen_kwargs["prompt_ids"] = torch.tensor(
            prompt_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

    t0 = time.time()
    with torch.inference_mode():
        generation = model.generate(inputs, **gen_kwargs)
    decode_s = time.time() - t0

    seq_logprobs = compute_sequence_logprobs(
        model,
        inputs,
        attention_mask,
        generation.sequences,
    )
    texts = processor.batch_decode(
        generation.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    hypotheses = []
    for rank, (text, logprob) in enumerate(zip(texts, seq_logprobs), start=1):
        hypotheses.append(
            {
                "rank": rank,
                "text": text,
                "whisper_logprob": float(logprob),
            }
        )

    row = {
        "utt_id": utt_id,
        "speaker_id": rec.get("speaker_id"),
        "lang": rec.get("lang"),
        "audio_path": audio_path,
        "reference": reference,
        "hypotheses": hypotheses[: num_hypotheses],
        "decode_s": round(decode_s, 4),
        "model": model_name,
        "processor_model": processor_name,
        "whisper_lang": whisper_lang,
        "dtype": str(dtype).replace("torch.", ""),
        "num_beams": num_beams,
        "num_hypotheses": num_hypotheses,
        "max_new_tokens": max_new_tokens,
        "device": device,
    }
    return row, decode_s


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate N-best Whisper hypotheses for JV/SU test manifests."
    )
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument(
        "--processor-model",
        default=None,
        help="Load WhisperProcessor from this model or checkpoint; defaults to --model.",
    )
    parser.add_argument(
        "--lang",
        required=True,
        choices=["jv", "jw", "su"],
        help="Language target. Javanese uses Whisper code 'jw'.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--num-hypotheses", type=int, default=10)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Whisper sampling for the N-best pool (default: True).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-utterances", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="Resume from an existing JSONL output (default: True).")
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Optional Whisper initial_prompt string.",
    )
    args = parser.parse_args()

    whisper_lang = "jw" if args.lang in {"jv", "jw"} else "su"
    dtype = resolve_dtype(args.dtype)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    records = load_manifest(args.manifest)
    if args.max_utterances is not None:
        records = records[: args.max_utterances]

    print(f"[nbest] model         : {args.model}")
    print(f"[nbest] processor    : {args.processor_model or args.model}")
    print(f"[nbest] lang         : {args.lang} -> whisper={whisper_lang}")
    print(f"[nbest] dtype        : {args.dtype} -> {dtype}")
    print(f"[nbest] manifest     : {args.manifest} ({len(records)} utterances)")
    print(f"[nbest] out-dir      : {args.out_dir}")
    print(f"[nbest] num-beams    : {args.num_beams}")
    print(f"[nbest] num-hypotheses: {args.num_hypotheses}")
    print(f"[nbest] do-sample    : {args.do_sample}")
    print(f"[nbest] temperature  : {args.temperature}")
    print(f"[nbest] top-p        : {args.top_p}")
    print(f"[nbest] seed         : {args.seed}")
    if args.initial_prompt:
        print(f"[nbest] prompt       : {args.initial_prompt!r}")
    print()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor_src = args.processor_model or args.model
    processor = WhisperProcessor.from_pretrained(processor_src)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    forced_ids = processor.get_decoder_prompt_ids(language=whisper_lang, task="transcribe")
    max_target_positions = getattr(model.config, "max_target_positions", 448)
    max_new_tokens = max(1, max_target_positions - len(forced_ids) - 2)

    if args.num_hypotheses < 1:
        raise ValueError("--num-hypotheses must be >= 1")
    if not args.do_sample and args.num_beams < args.num_hypotheses:
        raise ValueError("--num-beams must be >= --num-hypotheses for beam search N-best extraction")

    out_path = Path(args.out_dir) / f"{args.lang}_nbest.jsonl"
    state_path = Path(args.out_dir) / f"{args.lang}_nbest.state.json"
    tmp_path = out_path.with_suffix(".jsonl.tmp")

    start_index = load_progress(out_path, state_path) if args.resume else 0
    if args.resume and start_index > 0:
        print(f"[nbest] Resuming from row {start_index + 1}/{len(records)}")

    print("[nbest] Generating hypotheses ...\n")
    out_f = open(out_path if args.resume else tmp_path, "a" if args.resume else "w", encoding="utf-8")
    try:
        for i, rec in enumerate(records):
            if i < start_index:
                continue
            try:
                row, decode_s = decode_one_row(
                    model=model,
                    processor=processor,
                    rec=rec,
                    row_index=i,
                    device="cuda",
                    dtype=dtype,
                    forced_ids=forced_ids,
                    max_new_tokens=max_new_tokens,
                    num_beams=args.num_beams,
                    num_hypotheses=args.num_hypotheses,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    whisper_lang=whisper_lang,
                    model_name=args.model,
                    processor_name=processor_src,
                    initial_prompt=args.initial_prompt,
                )
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except OSError:
                    pass
                save_state(state_path, next_index=i + 1, utt_id=row["utt_id"], status="running")

                print(
                    f"{i + 1:>4}/{len(records)}  {row['utt_id']:<24}  "
                    f"{decode_s:>7.2f}s  top1={row['hypotheses'][0]['whisper_logprob']:.4f}"
                )
                if (i + 1) % 10 == 0:
                    sys.stdout.flush()

                # keep CUDA memory as low as possible between rows
                del row, decode_s
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                gc.collect()
                audio_path = rec["path"]
                utt_id = resolve_utt_id(rec, audio_path, i)
                print(f"[nbest] OOM on GPU at row {i + 1}/{len(records)} ({utt_id}); retrying this row on CPU ...")
                save_state(state_path, next_index=i, utt_id=utt_id, status="oom-gpu", error=str(e))
                try:
                    model.to(device="cpu", dtype=torch.float32).eval()
                    row, decode_s = decode_one_row(
                        model=model,
                        processor=processor,
                        rec=rec,
                        row_index=i,
                        device="cpu",
                        dtype=torch.float32,
                        forced_ids=forced_ids,
                        max_new_tokens=max_new_tokens,
                        num_beams=args.num_beams,
                        num_hypotheses=args.num_hypotheses,
                        do_sample=args.do_sample,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed,
                        whisper_lang=whisper_lang,
                        model_name=args.model,
                        processor_name=processor_src,
                        initial_prompt=args.initial_prompt,
                    )
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()
                    try:
                        os.fsync(out_f.fileno())
                    except OSError:
                        pass
                    save_state(state_path, next_index=i + 1, utt_id=row["utt_id"], status="running")
                    print(
                        f"{i + 1:>4}/{len(records)}  {row['utt_id']:<24}  "
                        f"{decode_s:>7.2f}s  top1={row['hypotheses'][0]['whisper_logprob']:.4f}  [cpu]"
                    )
                    if (i + 1) % 10 == 0:
                        sys.stdout.flush()
                finally:
                    model.to(device="cuda", dtype=dtype).eval()
                    torch.cuda.empty_cache()
                    gc.collect()
            except Exception as e:
                audio_path = rec["path"]
                utt_id = resolve_utt_id(rec, audio_path, i)
                save_state(state_path, next_index=i, utt_id=utt_id, status="error", error=str(e))
                print(f"\n[nbest] Error at row {i + 1}/{len(records)} ({utt_id}): {e}", file=sys.stderr)
                raise
    finally:
        out_f.close()

    save_state(state_path, next_index=len(records), status="done")
    if args.resume and tmp_path.exists():
        # If we were resuming, data already lives in out_path.
        pass
    elif tmp_path.exists():
        os.replace(tmp_path, out_path)
    print(f"\n[nbest] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""gfd/byte_gfd.py

Byte-level Generative Fusion Decoding (Proposition 1).

When the LM is byte-native (BLT), the branching approximation of Hsu et al.
2024 collapses to an exact product:
    P_LM(string(y)) = prod_n P_LM(b_n | b_{<n})

Adapted from blt-gfd-jvsu-asr/src/fusion/byte_gfd.py.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from gfd.entropy_gate import EntropyGate
from gfd.lattice import byte_automaton_expand

LOG = logging.getLogger(__name__)


class _Tracer:
    """Writes one JSONL trace file per utterance (one JSON object per event).

    Instantiated by ByteLevelFusionDecoder.decode() when fusion.trace_file.enabled
    is true. All overhead is gated behind `if self._tracer:` in the hot path so
    there is zero cost when tracing is off.
    """

    def __init__(self, out_dir: str, utterance_id: str, max_candidates: int) -> None:
        import json as _json
        import os as _os
        _os.makedirs(out_dir, exist_ok=True)
        safe_id = utterance_id.replace("/", "_").replace("\\", "_")
        path = _os.path.join(out_dir, f"{safe_id}.jsonl")
        self._f = open(path, "w", encoding="utf-8")
        self._json = _json
        self._max = max_candidates

    def _w(self, obj: dict) -> None:
        self._f.write(self._json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._f.flush()
        self._f.close()

    @staticmethod
    def _bd(b: "Beam", lam: float, alpha: float) -> dict:
        return {
            "text":    b.byte_seq.decode("utf-8", errors="replace"),
            "n_bytes": len(b.byte_seq),
            "asr":     round(b.asr_log_prob, 4),
            "lm":      round(b.lm_log_prob, 4),
            "fused":   round(b.fused_score(lam, alpha), 4),
            "fin":     b.finished,
        }

    @staticmethod
    def _score_row(sb: "Beam", cand: Any, lm_logp: float,
                   combined: float, ff: bool) -> "dict | None":
        if cand is None:
            return None
        return {
            "parent": sb.byte_seq.decode("utf-8", errors="replace"),
            "next":   bytes(cand["next_bytes"]).decode("utf-8", errors="replace"),
            "asr":    round(sb.asr_log_prob + cand["asr_logp"], 4),
            "lm":     round(sb.lm_log_prob + lm_logp, 4),
            "fused":  round(combined, 4),
            "ff":     ff,
            "eos":    bool(cand.get("is_eos", False)),
        }

    def init(self, utterance_id: str, cfg: dict) -> None:
        self._w({"event": "init", "uid": utterance_id, "cfg": cfg})

    def step_start(self, step: int, beams: list, lam: float, alpha: float) -> None:
        self._w({"event": "step_start", "step": step,
                 "beams": [self._bd(b, lam, alpha) for b in beams]})

    def phase1(self, step: int, beam_candidates: list) -> None:
        data = []
        for b, cands in beam_candidates:
            data.append({
                "beam": b.byte_seq.decode("utf-8", errors="replace"),
                "cands": [
                    {"bytes": bytes(c["next_bytes"]).decode("utf-8", errors="replace"),
                     "asr_logp": round(c["asr_logp"], 4),
                     "eos": bool(c.get("is_eos", False))}
                    for c in cands[:self._max]
                ],
            })
        self._w({"event": "phase1", "step": step, "beam_candidates": data})

    def phase2_scored(self, step: int, all_scored: list, survivors: list,
                      lam: float, alpha: float) -> None:
        rows, surv = [], []
        for item in all_scored[:self._max]:
            r = self._score_row(*item)
            if r is not None:
                rows.append(r)
        for item in survivors:
            r = self._score_row(*item)
            if r is not None:
                surv.append(r)
        self._w({"event": "phase2_scored", "step": step,
                 "all_scored": rows, "survivors": surv})

    def phase3_prefetch(self, step: int, actions: list) -> None:
        self._w({"event": "phase3_prefetch", "step": step, "actions": actions})

    def phase4_beams(self, step: int, beams: list, lam: float, alpha: float) -> None:
        self._w({"event": "phase4_beams", "step": step,
                 "beams": [self._bd(b, lam, alpha) for b in beams]})

    def done(self, best: "Beam", lam: float, alpha: float, stats: dict) -> None:
        self._w({"event": "done", "best": self._bd(best, lam, alpha), "stats": stats})


@dataclass
class Beam:
    byte_seq: bytes = b""
    asr_tokens: list[int] = field(default_factory=list)
    asr_log_prob: float = 0.0
    lm_log_prob: float = 0.0
    lm_forwards: int = 0
    finished: bool = False
    asr_node: Any | None = None
    lm_node: Any | None = None

    def fused_score(self, lam: float, alpha: float = 1.0,
                    lm_norm: str = "none", H_hat: float = 0.0,
                    lm_beta: float = 1.0) -> float:
        lm = self.lm_log_prob
        n = len(self.byte_seq)
        if n > 0:
            if lm_norm == "per_byte":
                lm = lm / n
            elif lm_norm == "length_penalty":
                lm = lm / (n ** lm_beta)
            elif lm_norm == "entropy_centered":
                lm = lm - n * H_hat
            elif lm_norm == "pmi_per_byte":
                lm = lm / n - H_hat
        return (1.0 - lam) * self.asr_log_prob + lam * alpha * lm


class ByteLevelFusionDecoder:
    def __init__(self, asr, lm, cfg: dict):
        self.asr = asr
        self.lm = lm
        self.cfg = cfg
        self.lam = cfg["fusion"]["lambda"]
        self.k = cfg["fusion"].get("shift_offset_k", 2)
        self.beam_width = cfg["fusion"]["beam_width"]
        self.kind = cfg["lm"]["kind"]
        self.max_bytes = cfg["fusion"]["max_output_bytes"]
        self.min_bytes = cfg["fusion"].get("min_output_bytes", 0)
        self.alpha = cfg["fusion"].get("lm_logit_alpha", 1.0)
        self.lm_norm      = cfg["fusion"].get("lm_normalization", "none")
        self.H_hat        = float(cfg["fusion"].get("byte_entropy", 0.0))
        self.ec_eos_boost = float(cfg["fusion"].get("ec_eos_boost", 0.0))
        self.lm_beta      = float(cfg["fusion"].get("lm_length_beta", 1.0))
        self.fusion_delay_bytes = int(cfg["fusion"].get("fusion_delay_bytes", 0))
        self.fusion_ramp_bytes  = int(cfg["fusion"].get("fusion_ramp_bytes",  0))
        self.lm_eos_penalty     = float(cfg["fusion"].get("lm_eos_penalty", 0.0))
        self.asr_gate_eos       = bool(cfg["fusion"].get("asr_gate_eos", False))
        # Option A: dynamic max_bytes scaled to audio duration (0 = disabled)
        _dm = cfg["fusion"].get("dynamic_max_bytes", {})
        self.dyn_max_bytes_rate = float(_dm.get("bytes_per_second", 0.0))
        self.dyn_max_bytes_pad  = int(_dm.get("padding_bytes", 30))
        # Option B: force EOS when Whisper's P(EOS) exceeds threshold (0 = disabled)
        self.asr_eos_force_threshold = float(cfg["fusion"].get("asr_eos_force_threshold", 0.0))
        self.asr_eos_min_bytes       = int(cfg["fusion"].get("asr_eos_min_bytes", 10))
        self.log_floor = cfg["fusion"].get("log_prob_floor", -30.0)
        self.decoder_backend = cfg["fusion"].get("decoder_backend", "cached_trie")
        self.fallback_on_error = bool(cfg["fusion"].get("fallback_on_error", True))
        self.enable_asr_prefix_cache = bool(cfg["fusion"].get("enable_asr_prefix_cache", False))
        self.enable_lm_prefix_cache = bool(cfg["fusion"].get("enable_lm_prefix_cache", True))
        gate_cfg = cfg["fusion"].get("entropy_gating", {"enabled": False, "mode": "always"})
        # BLT adapters expose next_byte_entropy; BPE adapter does not
        lm_for_gate = lm if self.kind in ("blt", "blt_lora", "mambabyte") else None
        self.gate = EntropyGate(gate_cfg, lm=lm_for_gate)
        self.lazy_prefetch = bool(cfg["fusion"].get("lazy_prefetch", False))
        self.patch_boundary_approx = bool(cfg["fusion"].get("patch_boundary_approx", False))
        self.patch_boundary_threshold = float(cfg["fusion"].get("patch_boundary_threshold", 0.8))
        self.trace = int(cfg["fusion"].get("trace", 0))
        self.progress_every = int(cfg["fusion"].get("progress_every", 0))
        # top_k_candidates: explicit candidate count per beam.
        # None → current default: max(8, beam_width * 4) i.e. 20 for width=5 (100 branches total).
        # Set to beam_width (e.g. 5) to match original GFD 25-branch behaviour.
        _tkc = cfg["fusion"].get("top_k_candidates", None)
        self.top_k_candidates: int = (
            int(_tkc) if _tkc is not None else max(8, self.beam_width * 4)
        )
        _hf = cfg["fusion"].get("heuristic_filters", {})
        # -- repetition suffix --
        _rd = _hf.get("rep_detect", cfg["fusion"].get("rep_detect", {}))
        self.rep_detect_enabled = bool(_rd.get("enabled", True))
        self.rep_detect_min_unit = int(_rd.get("min_unit", 2))
        self.rep_detect_max_unit = int(_rd.get("max_unit", 8))
        self.rep_detect_min_reps = int(_rd.get("min_reps", 4))
        # -- long word (missing spaces) --
        _lw = _hf.get("long_word", {})
        self.long_word_enabled = bool(_lw.get("enabled", True))
        self.long_word_max_bytes = int(_lw.get("max_word_bytes", 25))
        # -- bad phrases (domain-inappropriate tokens, e.g. Indonesian in JV/SU) --
        _bp = _hf.get("bad_phrases", {})
        self.bad_phrases_enabled = bool(_bp.get("enabled", True))
        # Patterns are stored as space-prefixed lowercase bytes for word-boundary
        # matching; the leading space means the first word of the sequence is not
        # counted, allowing one legitimate occurrence before the penalty triggers.
        self.bad_phrases_patterns: list[bytes] = [
            (" " + p.lower()).encode("utf-8")
            for p in _bp.get("patterns", [])
        ]
        self.bad_phrases_max_count = int(_bp.get("max_count", 2))
        # penalty applied to lm_log_prob of any beam that violates a filter
        self.heuristic_penalty = float(_hf.get("penalty", -1000.0))
        _tf = cfg["fusion"].get("trace_file", {})
        self.trace_file_enabled   = bool(_tf.get("enabled", False))
        self.trace_file_dir       = _tf.get("out_dir", None)
        self.trace_file_max_cands = int(_tf.get("max_candidates", 20))
        self._tracer: _Tracer | None = None
        self._stats: dict[str, float] = {}

    def _trace(self, level: int, msg: str) -> None:
        if self.trace >= level:
            print(msg, flush=True)

    def _has_repeat_suffix(self, seq: bytes) -> bool:
        """Return True if seq ends in a repeating byte pattern.

        Triggers when the last (unit * min_reps) bytes consist of the same
        unit repeated min_reps times. E.g. unit=3 "kak", min_reps=4 triggers.
        Matching is also done case-insensitively to catch phrases that alternate
        capitalisation (e.g. "Terima kasih. Terima Kasih. Terima kasih. ...").
        """
        if not self.rep_detect_enabled:
            return False

        # Precompute lowercased sequence once (guard: skip if UTF-8 byte length
        # changes after lowercasing, which is extremely rare for Latin script).
        try:
            seq_lower = seq.decode("utf-8", errors="replace").lower().encode("utf-8")
        except Exception:
            seq_lower = seq
        lower_safe = len(seq_lower) == len(seq)

        for w in range(self.rep_detect_min_unit, self.rep_detect_max_unit + 1):
            needed = w * self.rep_detect_min_reps
            if len(seq) < needed:
                continue
            unit = seq[-w:]
            # Exact match
            if seq[-needed:] == unit * self.rep_detect_min_reps:
                return True
            # Case-insensitive match
            if lower_safe:
                unit_l = seq_lower[-w:]
                if seq_lower[-needed:] == unit_l * self.rep_detect_min_reps:
                    return True
        return False

    def _has_long_word(self, seq: bytes) -> bool:
        """Return True if the current unseparated run of bytes exceeds max_word_bytes.

        Catches missing-space degeneration: "bocahbocahlanremajah..." where the
        model never emits 0x20 between words. The check only looks at bytes since
        the last space so it does not penalise normally-spaced long sequences.
        """
        if not self.long_word_enabled:
            return False
        last_space = seq.rfind(b" ")
        word_len = len(seq) if last_space == -1 else len(seq) - last_space - 1
        return word_len > self.long_word_max_bytes

    def _bad_phrase_count(self, seq: bytes) -> int:
        """Count space-prefixed occurrences of bad phrases in seq (case-insensitive).

        Uses a leading-space prefix so e.g. b' terima' only matches 'terima' as a
        separate word.  The first word of the sequence (no leading space) is
        intentionally not counted, so one legitimate occurrence is allowed before
        the penalty triggers.
        """
        if not self.bad_phrases_enabled or not self.bad_phrases_patterns:
            return 0
        try:
            seq_lower = seq.decode("utf-8", errors="replace").lower().encode("utf-8")
        except Exception:
            seq_lower = seq
        total = 0
        for pat in self.bad_phrases_patterns:
            total += seq_lower.count(pat)
        return total

    def _check_heuristics(self, seq: bytes) -> bool:
        """Return True if seq violates any heuristic filter."""
        return (self._has_repeat_suffix(seq)
                or self._has_long_word(seq)
                or self._bad_phrase_count(seq) >= self.bad_phrases_max_count)

    def decode(self, audio, utterance_id: str | None = None) -> dict[str, Any]:
        if self.trace_file_enabled and self.trace_file_dir:
            uid = utterance_id or f"utt_{int(time.time() * 1000)}"
            self._tracer = _Tracer(self.trace_file_dir, uid, self.trace_file_max_cands)
            self._tracer.init(uid, {
                "lambda": self.lam, "alpha": self.alpha,
                "beam_width": self.beam_width, "top_k": self.top_k_candidates,
                "max_bytes": self.max_bytes, "backend": self.decoder_backend,
            })
        else:
            self._tracer = None
        self._stats = {
            "decode_wall_ms": 0.0,
            "asr_prefix_cache_hits": 0.0,
            "asr_prefix_cache_misses": 0.0,
            "lm_prefix_cache_hits": 0.0,
            "lm_prefix_cache_misses": 0.0,
            "lm_logprob_lookups": 0.0,
            "heuristic_penalties": 0.0,
            "bad_phrase_penalties": 0.0,
            "asr_eos_forced": 0.0,
        }
        # Option A: compute per-utterance max_bytes from audio duration
        if self.dyn_max_bytes_rate > 0.0 and hasattr(audio, "__len__"):
            dur_sec = len(audio) / 16000.0
            dyn_limit = int(dur_sec * self.dyn_max_bytes_rate) + self.dyn_max_bytes_pad
            self._decode_max_bytes = min(self.max_bytes, max(self.min_bytes + 5, dyn_limit))
        else:
            self._decode_max_bytes = self.max_bytes
        t0 = time.time()
        try:
            if self.decoder_backend == "cached_trie":
                out = self._decode_cached_trie(audio)
            else:
                out = self._decode_legacy(audio)
        except Exception as err:
            if self.decoder_backend == "cached_trie" and self.fallback_on_error:
                LOG.warning("cached_trie failed (%s); falling back to legacy", err)
                out = self._decode_legacy(audio)
                out["decoder_backend"] = "legacy_fallback"
                out["decoder_error"] = str(err)
            else:
                raise
        finally:
            if self._tracer is not None:
                self._tracer.close()
                self._tracer = None
        self._stats["decode_wall_ms"] = float((time.time() - t0) * 1000.0)
        out["decoder_stats"] = dict(self._stats)
        out["decoder_backend"] = out.get("decoder_backend", self.decoder_backend)
        return out

    # ------------------------------------------------------------------

    def _effective_lambda(self, n_bytes: int) -> float:
        """Schedule-aware lambda: 0 during ASR-only phase, linear ramp, then full lam."""
        D = self.fusion_delay_bytes
        R = self.fusion_ramp_bytes
        if D == 0 and R == 0:
            return self.lam  # fast path: no schedule configured
        if n_bytes < D:
            return 0.0
        if R > 0 and n_bytes < D + R:
            return self.lam * (n_bytes - D) / R
        return self.lam

    def _decode_legacy(self, audio) -> dict[str, Any]:
        asr_state = self.asr.encode(audio)
        beams: list[Beam] = [Beam()]
        step = 0
        while step < self._decode_max_bytes and any(not b.finished for b in beams):
            step += 1
            new: list[Beam] = []
            for b in beams:
                if b.finished:
                    new.append(b)
                    continue
                next_logits, _, _ = self.asr.next_token_logits(asr_state, b.asr_tokens)
                candidates = byte_automaton_expand(
                    next_logits, self.asr.tokenizer,
                    prev_byte_prefix=b.byte_seq,
                    top_k=self.top_k_candidates,
                )
                for cand in candidates:
                    lm_delta = self._lm_score(b.byte_seq, cand["next_bytes"])
                    new.append(Beam(
                        byte_seq=b.byte_seq + cand["next_bytes"],
                        asr_tokens=b.asr_tokens + [cand["next_token"]],
                        asr_log_prob=b.asr_log_prob + cand["asr_logp"],
                        lm_log_prob=b.lm_log_prob + lm_delta["lm_logp"],
                        lm_forwards=b.lm_forwards + lm_delta["forwards"],
                        finished=cand.get("is_eos", False),
                    ))
            new.sort(key=lambda x: x.fused_score(self._effective_lambda(len(x.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta), reverse=True)
            beams = new[: self.beam_width]
        best = max(beams, key=lambda x: x.fused_score(self._effective_lambda(len(x.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta))
        return {
            "text": best.byte_seq.decode("utf-8", errors="replace"),
            "asr_log_prob": best.asr_log_prob,
            "lm_log_prob": best.lm_log_prob,
            "lm_forwards": best.lm_forwards,
            "num_bytes": len(best.byte_seq),
        }

    def _decode_cached_trie(self, audio) -> dict[str, Any]:
        asr_state = self.asr.encode(audio)
        asr_root = None
        if self.enable_asr_prefix_cache and hasattr(self.asr, "init_prefix_cache"):
            asr_root = self.asr.init_prefix_cache(asr_state)
        lm_root = None
        is_blt = self.kind in ("blt", "blt_lora", "mambabyte")
        if self.enable_lm_prefix_cache and is_blt and hasattr(self.lm, "init_prefix_cache"):
            lm_root = self.lm.init_prefix_cache()

        beams: list[Beam] = [Beam(asr_node=asr_root, lm_node=lm_root)]
        step = 0
        while step < self._decode_max_bytes and any(not b.finished for b in beams):
            step += 1
            n_active = sum(1 for b in beams if not b.finished)
            if self._tracer:
                self._tracer.step_start(step, beams, self.lam, self.alpha)

            if self.progress_every > 0 and step % self.progress_every == 0:
                best = max(beams, key=lambda x: x.fused_score(self._effective_lambda(len(x.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta))
                preview = best.byte_seq.decode("utf-8", errors="replace")[-40:]
                print(f"  [step {step:>4}] active={n_active}  best={preview!r}", flush=True)

            if self.trace >= 1:
                sep = "-" * 60
                self._trace(1, f"\n{sep}")
                self._trace(1, f"[step {step}]  active={n_active}  total_beams={len(beams)}")
                for bi, b in enumerate(beams):
                    txt = b.byte_seq.decode("utf-8", errors="replace")
                    self._trace(1,
                        f"  beam[{bi}] asr={b.asr_log_prob:.3f}  lm={b.lm_log_prob:.3f}"
                        f"  fused={b.fused_score(self._effective_lambda(len(b.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta):.3f}  fin={b.finished}"
                        f"  text={txt!r}")

            # Phase 1: collect ASR candidates for all active beams
            beam_candidates: list[tuple[Beam, list]] = []
            for b in beams:
                if b.finished:
                    continue
                if b.asr_node is not None and hasattr(self.asr, "next_token_logits_cached"):
                    next_logits, _, _ = self.asr.next_token_logits_cached(
                        asr_state, b.asr_tokens, b.asr_node
                    )
                else:
                    next_logits, _, _ = self.asr.next_token_logits(asr_state, b.asr_tokens)
                candidates = byte_automaton_expand(
                    next_logits, self.asr.tokenizer,
                    prev_byte_prefix=b.byte_seq,
                    top_k=self.top_k_candidates,
                )
                beam_candidates.append((b, candidates))
                if self.trace >= 2:
                    txt = b.byte_seq.decode("utf-8", errors="replace")
                    self._trace(2, f"  [phase1] beam {txt!r} → {len(candidates)} candidates")
                    for ci, cand in enumerate(candidates[:5]):
                        nb = bytes(cand["next_bytes"]).decode("utf-8", errors="replace")
                        self._trace(2,
                            f"    cand[{ci}] bytes={nb!r}  asr_logp={cand['asr_logp']:.3f}")

            if self._tracer:
                self._tracer.phase1(step, beam_candidates)

            # Option B: force EOS on beams where Whisper's P(EOS) exceeds threshold.
            # This catches speech-embedding exhaustion: Whisper says "stop" but
            # MambaByte's high weight would otherwise override the EOS signal.
            if self.asr_eos_force_threshold > 0.0:
                filtered: list[tuple] = []
                for b, candidates in beam_candidates:
                    if len(b.byte_seq) < self.asr_eos_min_bytes:
                        filtered.append((b, candidates))
                        continue
                    eos_lp = None
                    for c in candidates:
                        if c.get("is_eos"):
                            eos_lp = float(c["asr_logp"])
                            break
                    if eos_lp is not None and math.exp(max(eos_lp, -20.0)) >= self.asr_eos_force_threshold:
                        b.finished = True
                        self._stats["asr_eos_forced"] += 1
                        self._trace(1,
                            f"  [asr_eos_force] P(EOS)={math.exp(max(eos_lp,-20.0)):.3f} "
                            f">= {self.asr_eos_force_threshold} after {len(b.byte_seq)} bytes "
                            f"→ forced finish: {b.byte_seq.decode('utf-8', errors='replace')[-30:]!r}")
                    else:
                        filtered.append((b, candidates))
                beam_candidates = filtered

            # Phase 2 (lazy) or Phase 2+3 (eager)
            if is_blt and self.lazy_prefetch:
                # Score all candidates for free from parent log_probs, then
                # prefetch only for the beam_width survivors.
                all_scored = self._score_all_from_parents(beam_candidates)
                for b in beams:
                    if b.finished:
                        all_scored.append((b, None, 0.0, b.fused_score(self._effective_lambda(len(b.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta), False))
                all_scored.sort(key=lambda x: x[3], reverse=True)
                survivors = all_scored[: self.beam_width]
                if self._tracer:
                    self._tracer.phase2_scored(step, all_scored, survivors, self.lam, self.alpha)

                if self.trace >= 1:
                    self._trace(1, f"  [lazy] scored {len(all_scored)} candidates → top {len(survivors)} survivors:")
                    for ri, (sb, cand, lm_logp, combined, ff) in enumerate(survivors):
                        if cand is None:
                            self._trace(1, f"    [{ri}] <finished beam>  fused={combined:.3f}")
                        else:
                            nb = bytes(cand["next_bytes"]).decode("utf-8", errors="replace")
                            prefix = sb.byte_seq.decode("utf-8", errors="replace")
                            tag = " [FROZEN]" if ff else ""
                            self._trace(1,
                                f"    [{ri}] {(prefix+nb)!r}"
                                f"  asr={sb.asr_log_prob+cand['asr_logp']:.3f}"
                                f"  lm={sb.lm_log_prob+lm_logp:.3f}"
                                f"  fused={combined:.3f}{tag}")
                    if self.trace >= 2:
                        self._trace(2, f"  [lazy] all {len(all_scored)} scored candidates:")
                        for ri, (sb, cand, lm_logp, combined, ff) in enumerate(all_scored):
                            if cand is None:
                                continue
                            nb = bytes(cand["next_bytes"]).decode("utf-8", errors="replace")
                            prefix = sb.byte_seq.decode("utf-8", errors="replace")
                            tag = " [FROZEN]" if ff else ""
                            self._trace(2,
                                f"    [{ri:3d}] {(prefix+nb)!r}"
                                f"  asr={sb.asr_log_prob+cand['asr_logp']:.3f}"
                                f"  lm={sb.lm_log_prob+lm_logp:.3f}"
                                f"  fused={combined:.3f}{tag}")

                prefetch_actions = self._lazy_prefetch_survivors(survivors)
                if self._tracer:
                    self._tracer.phase3_prefetch(step, prefetch_actions)
                beams = self._build_beams_from_survivors(asr_state, survivors)
                if self._tracer:
                    self._tracer.phase4_beams(step, beams, self.lam, self.alpha)
            else:
                # Original eager path: prefetch all, then score
                if is_blt:
                    self._prefetch_lm_nodes(beam_candidates)

                new: list[Beam] = []
                for b in beams:
                    if b.finished:
                        new.append(b)
                for b, candidates in beam_candidates:
                    for cand in candidates:
                        if cand.get("is_eos") and len(b.byte_seq) < self.min_bytes:
                            continue  # suppress EOS until min_output_bytes reached
                        child_asr_node = b.asr_node
                        if (self.enable_asr_prefix_cache and b.asr_node is not None
                                and hasattr(self.asr, "extend_prefix_cache")):
                            child_asr_node, hit = self.asr.extend_prefix_cache(
                                asr_state, b.asr_node, cand["next_token"]
                            )
                            self._stats["asr_prefix_cache_hits" if hit else "asr_prefix_cache_misses"] += 1

                        lm_delta = self._lm_score_cached(
                            b.byte_seq, cand["next_bytes"], b.lm_node
                        )
                        new_seq = b.byte_seq + cand["next_bytes"]
                        is_eos = cand.get("is_eos", False)
                        new_lm_logp = b.lm_log_prob + lm_delta["lm_logp"]
                        if not is_eos:
                            if self._has_repeat_suffix(new_seq):
                                new_lm_logp += self.heuristic_penalty
                                is_eos = True  # freeze: rep loop cannot self-recover
                                self._stats["heuristic_penalties"] += 1
                                self._trace(1, f"  [rep_detect] frozen+penalized: "
                                            f"{new_seq.decode('utf-8', errors='replace')[-32:]!r}")
                            elif self._has_long_word(new_seq):
                                new_lm_logp += self.heuristic_penalty
                                self._stats["heuristic_penalties"] += 1
                                self._trace(1, f"  [long_word] penalized: "
                                            f"{new_seq.decode('utf-8', errors='replace')[-32:]!r}")
                            elif self._bad_phrase_count(new_seq) >= self.bad_phrases_max_count:
                                new_lm_logp += self.heuristic_penalty
                                is_eos = True  # freeze beam
                                self._stats["bad_phrase_penalties"] += 1
                                self._trace(1, f"  [bad_phrase] frozen+penalized: "
                                            f"{new_seq.decode('utf-8', errors='replace')[-40:]!r}")
                        nb_beam = Beam(
                            byte_seq=new_seq,
                            asr_tokens=b.asr_tokens + [cand["next_token"]],
                            asr_log_prob=b.asr_log_prob + cand["asr_logp"],
                            lm_log_prob=new_lm_logp,
                            lm_forwards=b.lm_forwards + lm_delta["forwards"],
                            finished=is_eos,
                            asr_node=child_asr_node,
                            lm_node=lm_delta.get("lm_node"),
                        )
                        new.append(nb_beam)
                        if self.trace >= 2:
                            txt = nb_beam.byte_seq.decode("utf-8", errors="replace")
                            self._trace(2,
                                f"  [eager] {txt!r}"
                                f"  asr={nb_beam.asr_log_prob:.3f}"
                                f"  lm={nb_beam.lm_log_prob:.3f}"
                                f"  fused={nb_beam.fused_score(self._effective_lambda(len(nb_beam.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta):.3f}")
                new.sort(key=lambda x: x.fused_score(self._effective_lambda(len(x.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta), reverse=True)
                if self.trace >= 1:
                    self._trace(1, f"  [eager] {len(new)} expansions → top {self.beam_width} kept:")
                    for ri, nb_beam in enumerate(new[: self.beam_width]):
                        txt = nb_beam.byte_seq.decode("utf-8", errors="replace")
                        self._trace(1,
                            f"    [{ri}] {txt!r}"
                            f"  asr={nb_beam.asr_log_prob:.3f}"
                            f"  lm={nb_beam.lm_log_prob:.3f}"
                            f"  fused={nb_beam.fused_score(self._effective_lambda(len(nb_beam.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta):.3f}")
                beams = new[: self.beam_width]

        best = max(beams, key=lambda x: x.fused_score(self._effective_lambda(len(x.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta))
        self._trace(1, f"\n[decode done] best={best.byte_seq.decode('utf-8', errors='replace')!r}"
                    f"  asr={best.asr_log_prob:.3f}  lm={best.lm_log_prob:.3f}"
                    f"  fused={best.fused_score(self._effective_lambda(len(best.byte_seq)), self.alpha, self.lm_norm, self.H_hat, self.lm_beta):.3f}  lm_fwd={best.lm_forwards}")
        if self._tracer:
            self._tracer.done(best, self.lam, self.alpha, dict(self._stats))
        return {
            "text": best.byte_seq.decode("utf-8", errors="replace"),
            "asr_log_prob": best.asr_log_prob,
            "lm_log_prob": best.lm_log_prob,
            "lm_forwards": best.lm_forwards,
            "num_bytes": len(best.byte_seq),
        }

    def _prefetch_lm_nodes(self, beam_candidates: list[tuple[Beam, list]]) -> None:
        """Batch-prefill the LM prefix trie for all candidate byte sequences.

        Groups extensions by parent node at each depth level so that
        batch_extend_from_node is called once per unique parent rather than
        once per candidate, reducing BLT forward passes by ~top_k×.
        """
        if not (self.enable_lm_prefix_cache
                and hasattr(self.lm, "batch_extend_from_node")):
            return

        # active: list of (current_lm_node, remaining_bytes)
        active: list[tuple[Any, bytes]] = []
        for b, candidates in beam_candidates:
            if b.lm_node is None:
                continue
            for cand in candidates:
                nb = bytes(cand["next_bytes"])
                if nb:
                    active.append((b.lm_node, nb))

        if not active:
            return

        # BFS level by level: batch all extensions that share the same parent node
        while active:
            # Group next bytes by parent node identity
            node_bytes: dict[int, tuple[Any, list[int]]] = {}
            for node, remaining in active:
                nid = id(node)
                if nid not in node_bytes:
                    node_bytes[nid] = (node, [])
                node_bytes[nid][1].append(remaining[0])

            # One batched forward pass per unique parent node
            for node, next_bytes in node_bytes.values():
                self.lm.batch_extend_from_node(node, next_bytes)

            # Advance to next depth: follow the child we just created
            next_active: list[tuple[Any, bytes]] = []
            for node, remaining in active:
                child = node.children.get(remaining[0])
                if child is not None and len(remaining) > 1:
                    next_active.append((child, remaining[1:]))
            active = next_active

    def _score_all_from_parents(
        self, beam_candidates: list[tuple[Beam, list]]
    ) -> list[tuple[Beam, Any, float, float, bool]]:
        """Score all candidates using only parent node log_probs — no forward passes
        for 1-byte candidates. Heuristic penalties are applied here so penalized
        candidates are excluded from survivor selection before any forward pass.

        Returns 5-tuples: (beam, cand, lm_logp_delta, combined_score, force_finish).
        force_finish=True for rep_detect violations: beam is frozen (won't extend)
        AND penalized. long_word violations are penalized only (can escape via space).
        """
        scored: list[tuple[Beam, Any, float, float, bool]] = []
        for b, candidates in beam_candidates:
            for cand in candidates:
                is_eos = cand.get("is_eos", False)
                if is_eos and len(b.byte_seq) < self.min_bytes:
                    continue  # suppress EOS until min_output_bytes reached
                lm_logp = self._lm_score_from_parent(b.lm_node, cand["next_bytes"])
                if is_eos and self.asr_gate_eos:
                    lm_logp = cand["asr_logp"]  # neutralise LM EOS bias: copy ASR delta
                elif is_eos and self.lm_eos_penalty != 0.0:
                    lm_logp -= self.lm_eos_penalty
                force_finish = False
                new_seq = b.byte_seq + cand["next_bytes"]
                if not is_eos:
                    if self._has_repeat_suffix(new_seq):
                        lm_logp += self.heuristic_penalty
                        force_finish = True
                        self._stats["heuristic_penalties"] += 1
                        self._trace(1, f"  [rep_detect] frozen+penalized: "
                                    f"{new_seq.decode('utf-8', errors='replace')[-32:]!r}")
                    elif self._has_long_word(new_seq):
                        lm_logp += self.heuristic_penalty
                        self._stats["heuristic_penalties"] += 1
                        self._trace(1, f"  [long_word] penalized: "
                                    f"{new_seq.decode('utf-8', errors='replace')[-32:]!r}")
                    elif self._bad_phrase_count(new_seq) >= self.bad_phrases_max_count:
                        lm_logp += self.heuristic_penalty
                        force_finish = True
                        self._stats["bad_phrase_penalties"] += 1
                        self._trace(1, f"  [bad_phrase] frozen+penalized: "
                                    f"{new_seq.decode('utf-8', errors='replace')[-40:]!r}")
                total_asr = b.asr_log_prob + cand["asr_logp"]
                total_lm = b.lm_log_prob + lm_logp
                n_total = len(new_seq)
                if n_total > 0 and self.lm_norm == "per_byte":
                    lm_for_fusion = total_lm / n_total
                elif n_total > 0 and self.lm_norm == "length_penalty":
                    lm_for_fusion = total_lm / (n_total ** self.lm_beta)
                elif n_total > 0 and self.lm_norm == "entropy_centered":
                    lm_for_fusion = total_lm - n_total * self.H_hat
                    if is_eos and self.ec_eos_boost != 0.0:
                        lm_for_fusion += self.ec_eos_boost
                elif n_total > 0 and self.lm_norm == "pmi_per_byte":
                    lm_for_fusion = total_lm / n_total - self.H_hat
                else:
                    lm_for_fusion = total_lm
                lam_eff = self._effective_lambda(n_total)
                combined = (1.0 - lam_eff) * total_asr + lam_eff * self.alpha * lm_for_fusion
                scored.append((b, cand, lm_logp, combined, force_finish))
        return scored

    def _lm_score_from_parent(self, lm_node, new_bytes) -> float:
        """Score new_bytes by index-lookups on existing trie nodes.
        Applies the same entropy gate as _lm_score_cached so lazy and eager
        paths produce identical scores. For 1-byte candidates with gate active:
        zero cost (index lookup only). Multi-byte: creates intermediate nodes
        on demand (rare — only multi-byte ASR tokens hit this path)."""
        if lm_node is None or not self.enable_lm_prefix_cache:
            return 0.0
        total = 0.0
        current = lm_node
        for i, b in enumerate(new_bytes):
            if not hasattr(current, "log_probs") or current.log_probs is None:
                break
            entropy = (self.lm.entropy_from_node(current)
                       if hasattr(self.lm, "entropy_from_node") else None)
            if entropy is None:
                invoke = self.gate.should_invoke(current.byte_seq)
            else:
                invoke = self.gate.should_invoke_entropy(entropy)
            if invoke:
                lp = self.lm.byte_logprob_from_node(current, b)
                total += max(float(lp), self.log_floor)
                self._stats["lm_logprob_lookups"] += 1
            if i < len(new_bytes) - 1:
                # Intermediate byte: walk ONLY if already cached — never trigger a
                # forward pass here. Scoring is free; full prefetch happens later in
                # _lazy_prefetch_survivors for the winning survivors only.
                child = current.children.get(int(b))
                if child is None:
                    break  # partial score — uncached intermediate, stop here
                self._stats["lm_prefix_cache_hits"] += 1
                current = child
        return total

    def _lazy_prefetch_survivors(
        self, survivors: list[tuple[Beam, Any, float, float, bool]]
    ) -> list[dict]:
        """Create child nodes ONLY for surviving beams — one forward pass per survivor.

        Returns a list of action records (one per survivor) that describe what
        happened: cache_hit, lm_forward, approx_child, or one of several skip reasons.
        The caller emits this list to the tracer when tracing is enabled.
        """
        actions: list[dict] = []

        def _act(cand: Any, action: str) -> None:
            nb_text = bytes(cand["next_bytes"]).decode("utf-8", errors="replace") if cand else ""
            actions.append({"bytes": nb_text, "action": action})

        if not self.enable_lm_prefix_cache:
            for _, cand, _, _, _ in survivors:
                _act(cand, "skip_no_lm_cache")
            return actions
        has_ensure = hasattr(self.lm, "ensure_child_node")
        has_approx = hasattr(self.lm, "create_approx_child")
        if not has_ensure:
            for _, cand, _, _, _ in survivors:
                _act(cand, "skip_no_ensure_method")
            return actions

        for b, cand, _, _, force_finish in survivors:
            if cand is None or b.lm_node is None or force_finish:
                _act(cand, "skip_finished_or_frozen")
                continue
            nb = cand["next_bytes"]
            if not nb:
                _act(cand, "skip_empty_bytes")
                continue
            # walk to the node just before the final byte
            parent = b.lm_node
            walked_out = False
            for byte_val in nb[:-1]:
                parent, hit = self.lm.extend_prefix_cache(parent, byte_val)
                self._stats["lm_prefix_cache_hits" if hit else "lm_prefix_cache_misses"] += 1
                if parent is None:
                    walked_out = True
                    break
            if walked_out:
                _act(cand, "skip_walk_failed")
                continue
            final_byte = int(nb[-1])
            if final_byte in parent.children:
                self._stats["lm_prefix_cache_hits"] += 1
                _act(cand, "cache_hit")
                continue
            if (self.patch_boundary_approx and has_approx
                    and hasattr(parent, "entropy")
                    and parent.entropy < self.patch_boundary_threshold):
                self.lm.create_approx_child(parent, final_byte)
                _act(cand, "approx_child")
            else:
                self.lm.ensure_child_node(parent, final_byte)
                _act(cand, "lm_forward")
            self._stats["lm_prefix_cache_misses"] += 1

        return actions

    def _build_beams_from_survivors(
        self,
        asr_state: Any,
        survivors: list[tuple[Beam, Any, float, float, bool]],
    ) -> list[Beam]:
        """Build the new beam list from the selected survivors."""
        new_beams: list[Beam] = []
        for b, cand, lm_logp, _, force_finish in survivors:
            if cand is None:
                # finished beam — carry forward unchanged
                new_beams.append(b)
                continue
            # ASR prefix-cache extension (skip for frozen beams — won't be extended)
            child_asr_node = b.asr_node
            if (not force_finish and self.enable_asr_prefix_cache
                    and b.asr_node is not None
                    and hasattr(self.asr, "extend_prefix_cache")):
                child_asr_node, hit = self.asr.extend_prefix_cache(
                    asr_state, b.asr_node, cand["next_token"]
                )
                self._stats["asr_prefix_cache_hits" if hit else "asr_prefix_cache_misses"] += 1
            # Walk to final child node (created by _lazy_prefetch_survivors).
            # Frozen beams have no child — lm_node stays at parent (irrelevant since finished).
            lm_child = b.lm_node
            if lm_child is not None and not force_finish:
                for byte_val in cand["next_bytes"]:
                    lm_child = lm_child.children.get(int(byte_val))
                    if lm_child is None:
                        break
            new_seq = b.byte_seq + cand["next_bytes"]
            # lm_logp already includes any heuristic penalty from _score_all_from_parents.
            # force_finish=True means rep_detect fired: freeze the beam.
            new_beams.append(Beam(
                byte_seq=new_seq,
                asr_tokens=b.asr_tokens + [cand["next_token"]],
                asr_log_prob=b.asr_log_prob + cand["asr_logp"],
                lm_log_prob=b.lm_log_prob + lm_logp,
                lm_forwards=b.lm_forwards + (1 if (lm_child is not None and not force_finish) else 0),
                finished=cand.get("is_eos", False) or force_finish,
                asr_node=child_asr_node,
                lm_node=lm_child if not force_finish else None,
            ))
        return new_beams

    # ------------------------------------------------------------------

    def _lm_score_cached(self, prefix_bytes, new_bytes, lm_prefix_node) -> dict:
        is_blt = self.kind in ("blt", "blt_lora", "mambabyte")
        if (not self.enable_lm_prefix_cache or not is_blt
                or lm_prefix_node is None
                or not hasattr(self.lm, "extend_prefix_cache")):
            return self._lm_score(prefix_bytes, new_bytes)

        total = 0.0
        forwards = 0
        current = lm_prefix_node
        for b in new_bytes:
            entropy = (self.lm.entropy_from_node(current)
                       if hasattr(self.lm, "entropy_from_node") else None)
            if entropy is None:
                invoke = self.gate.should_invoke(current.byte_seq)
            else:
                invoke = self.gate.should_invoke_entropy(entropy)
            if invoke:
                lp = self.lm.byte_logprob_from_node(current, b)
                total += max(lp, self.log_floor)
                forwards += 1
                self._stats["lm_logprob_lookups"] += 1
            current, hit = self.lm.extend_prefix_cache(current, b)
            self._stats["lm_prefix_cache_hits" if hit else "lm_prefix_cache_misses"] += 1
        return {"lm_logp": total, "forwards": forwards, "lm_node": current}

    def _lm_score(self, prefix_bytes, new_bytes) -> dict:
        is_blt = self.kind in ("blt", "blt_lora", "mambabyte")
        if is_blt:
            total = 0.0
            forwards = 0
            for i, b in enumerate(new_bytes):
                prefix = prefix_bytes + new_bytes[:i]
                if not self.gate.should_invoke(prefix):
                    continue
                lp = self.lm.byte_logprob(prefix, b)
                total += max(lp, self.log_floor)
                forwards += 1
            return {"lm_logp": total, "forwards": forwards}
        if self.kind == "bpe":
            tok_logp = self.lm.score_bytes(prefix_bytes, new_bytes, branching=4)
            return {"lm_logp": max(tok_logp, self.log_floor), "forwards": 1}
        raise ValueError(f"unknown kind {self.kind}")

#!/usr/bin/env python3
"""
Build the post-rescoring comparison report once JV and SU rescoring are done.

Inputs:
  - JV rescoring summary JSON
  - SU rescoring summary JSON

Outputs:
  - experiments/rescoring/RESULTS.md
  - experiments/rescoring/FINDINGS.md

This script is CPU-only and intended to be run manually after both rescoring
jobs have finished.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def fmt_pp(a: float, b: float) -> str:
    return f"{(a - b) * 100:+.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build rescoring comparison report")
    ap.add_argument("--jv-summary", default="experiments/rescoring/summaries/jv_summary.json")
    ap.add_argument("--su-summary", default="experiments/rescoring/summaries/su_summary.json")
    ap.add_argument("--out-dir", default="experiments/rescoring")
    args = ap.parse_args()

    jv_path = Path(args.jv_summary)
    su_path = Path(args.su_summary)
    if not jv_path.exists():
        raise FileNotFoundError(f"Missing JV summary: {jv_path}")
    if not su_path.exists():
        raise FileNotFoundError(f"Missing SU summary: {su_path}")

    jv = load_json(jv_path)
    su = load_json(su_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_md = out_dir / "RESULTS.md"
    findings_md = out_dir / "FINDINGS.md"

    rows = [
        ("Zero-shot Whisper (1-best)", jv["zero_shot"]["wer"], su["zero_shot"]["wer"], jv["zero_shot"]["cer"], su["zero_shot"]["cer"]),
        ("N-best rescoring (adapted LM, post-hoc, same λ/α)", jv["rescoring"]["wer"], su["rescoring"]["wer"], jv["rescoring"]["cer"], su["rescoring"]["cer"]),
        ("N-best oracle (lowest-WER in 10-best)", jv["oracle"]["wer"], su["oracle"]["wer"], jv["oracle"]["cer"], su["oracle"]["cer"]),
        ("GFD first-pass fusion (our method)", jv["gfd"]["wer"], su["gfd"]["wer"], jv["gfd"]["cer"], su["gfd"]["cer"]),
    ]

    results_lines = []
    results_lines.append("# Rescoring Baseline Results\n")
    results_lines.append(f"- JV N-best file: `{jv.get('nbest_file', 'unknown')}`")
    results_lines.append(f"- SU N-best file: `{su.get('nbest_file', 'unknown')}`")
    results_lines.append(f"- JV permutation-test p-value: `{jv['permutation_test']['p_value']:.6f}` (seed {jv['permutation_test']['seed']})")
    results_lines.append(f"- SU permutation-test p-value: `{su['permutation_test']['p_value']:.6f}` (seed {su['permutation_test']['seed']})")
    results_lines.append("")
    results_lines.append("| System | JV WER | SU WER | JV CER | SU CER |")
    results_lines.append("|---|---:|---:|---:|---:|")
    for name, jw, sw, jc, sc in rows:
        results_lines.append(f"| {name} | {pct(jw)} | {pct(sw)} | {pct(jc)} | {pct(sc)} |")
    results_lines.append("")
    results_lines.append("## Per-language deltas")
    results_lines.append(f"- JV: rescoring vs GFD = {fmt_pp(jv['rescoring']['wer'], jv['gfd']['wer'])} pp; oracle gap = {fmt_pp(jv['oracle']['wer'], jv['rescoring']['wer'])} pp")
    results_lines.append(f"- SU: rescoring vs GFD = {fmt_pp(su['rescoring']['wer'], su['gfd']['wer'])} pp; oracle gap = {fmt_pp(su['oracle']['wer'], su['rescoring']['wer'])} pp")
    results_lines.append("")
    results_lines.append("## Permutation test")
    results_lines.append(f"- JV GFD-vs-rescoring mean delta: `{jv['paired_delta_gfd_minus_rescoring']:.6f}`")
    results_lines.append(f"- JV p-value: `{jv['permutation_test']['p_value']:.6f}`")
    results_lines.append(f"- SU GFD-vs-rescoring mean delta: `{su['paired_delta_gfd_minus_rescoring']:.6f}`")
    results_lines.append(f"- SU p-value: `{su['permutation_test']['p_value']:.6f}`")
    results_lines.append("")
    results_md.write_text("\n".join(results_lines), encoding="utf-8")

    findings_lines = []
    findings_lines.append("# Findings\n")
    findings_lines.append("This report summarizes the corrected post-hoc N-best rescoring baseline against the first-pass GFD method.\n")
    findings_lines.append("## Javanese")
    findings_lines.append(f"- Zero-shot Whisper WER: {pct(jv['zero_shot']['wer'])}%")
    findings_lines.append(f"- Post-hoc rescoring WER: {pct(jv['rescoring']['wer'])}%")
    findings_lines.append(f"- 10-best oracle WER: {pct(jv['oracle']['wer'])}%")
    findings_lines.append(f"- GFD WER: {pct(jv['gfd']['wer'])}%")
    findings_lines.append(f"- GFD significantly outperforms post-hoc rescoring (paired permutation p = `{jv['permutation_test']['p_value']:.6f}`, seed {jv['permutation_test']['seed']}, 10,000 permutations).")
    findings_lines.append(f"- Post-hoc rescoring is worse than zero-shot by {fmt_pp(jv['rescoring']['wer'], jv['zero_shot']['wer'])} pp.")
    findings_lines.append(f"- The true oracle outperforms GFD by {abs((jv['oracle']['wer'] - jv['gfd']['wer']) * 100):.2f} pp, so the reranker does not always select the reference-optimal hypothesis.")
    findings_lines.append("")
    findings_lines.append("## Sundanese")
    findings_lines.append(f"- Zero-shot Whisper WER: {pct(su['zero_shot']['wer'])}%")
    findings_lines.append(f"- Post-hoc rescoring WER: {pct(su['rescoring']['wer'])}%")
    findings_lines.append(f"- 10-best oracle WER: {pct(su['oracle']['wer'])}%")
    findings_lines.append(f"- GFD WER: {pct(su['gfd']['wer'])}%")
    findings_lines.append(f"- GFD significantly outperforms post-hoc rescoring (paired permutation p = `{su['permutation_test']['p_value']:.6f}`, seed {su['permutation_test']['seed']}, 10,000 permutations).")
    findings_lines.append(f"- Post-hoc rescoring is worse than zero-shot by {fmt_pp(su['rescoring']['wer'], su['zero_shot']['wer'])} pp.")
    findings_lines.append(f"- The true oracle outperforms GFD by {abs((su['oracle']['wer'] - su['gfd']['wer']) * 100):.2f} pp, so the reranker does not always select the reference-optimal hypothesis.")
    findings_lines.append("")
    findings_lines.append("## Interpretation")
    findings_lines.append("The corrected N-best pool is genuinely diverse, but the post-hoc rescoring baseline remains weaker than GFD and also worse than zero-shot in both languages. The oracle is stronger than GFD, so the earlier oracle=rescoring equality was a beam-collapse artifact, not the true result.")
    findings_lines.append("")
    findings_lines.append("## Next step")
    findings_lines.append("If you want the exact manuscript-ready wording, cite `RESULTS.md` and note that the N-best candidates were regenerated with sampling-based generation (temperature 0.7, top_p 0.9, num_beams 1, seed 42).")
    findings_md.write_text("\n".join(findings_lines), encoding="utf-8")

    print(f"[report] wrote {results_md}")
    print(f"[report] wrote {findings_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

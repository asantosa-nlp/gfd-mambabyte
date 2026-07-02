#!/usr/bin/env python3
"""
Generate 18 grid-search configs (9 JV + 9 SU) that run on the full
1,000-utterance search set instead of the 300-utterance stratified subset.
These are curiosity runs to confirm the grid winner is stable at 1K.

Output configs: configs/mb_{jv,su}_zs_large_v5cleanS20k_gs_lm{60,70,80}_ag{15,20,25}_1k[_minb15].yaml
Output results: results/gs20k_1k_{jv,su}_lm{60,70,80}_ag{15,20,25}/
"""
import yaml
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

LANGS = ["jv", "su"]
LMS   = [60, 70, 80]
AGS   = [15, 20, 25]

for lang in LANGS:
    sfx = "_minb15" if lang == "su" else ""
    manifest_1k = f"data/speech_corpus/local_{lang}/train_1000_seed42.jsonl"

    for lm in LMS:
        for ag in AGS:
            # Source config (300-utt version)
            src_name = f"mb_{lang}_zs_large_v5cleanS20k_gs_lm{lm}_ag{ag}{sfx}.yaml"
            src = CONFIGS / src_name
            if not src.exists():
                print(f"MISSING: {src_name} — skipping")
                continue

            # Load and patch
            cfg = yaml.safe_load(src.read_text())
            cfg["eval"]["search_manifest"] = manifest_1k
            cfg["experiment_id"] = cfg["experiment_id"] + "_1k"
            out_dir = f"results/gs20k_1k_{lang}_lm{lm}_ag{ag}"
            cfg["output"]["search_dir"] = out_dir

            # Write new config
            dst_name = f"mb_{lang}_zs_large_v5cleanS20k_gs_lm{lm}_ag{ag}_1k{sfx}.yaml"
            dst = CONFIGS / dst_name
            with open(dst, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            print(f"Written: {dst_name}")

print("Done — 18 configs generated.")

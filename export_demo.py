"""Score a handful of well-known structures and write docs/predictions.json.

The demo page is a viewer, not an inference engine: predictions are computed
here with the trained checkpoint and shipped as JSON, so the page stays small
and needs no ML runtime.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from predict import fetch_pdb, predict

# Structures chosen to be recognisable and to cover different pocket types.
TARGETS = [
    ("1m17", "EGFR kinase + erlotinib", "EGFR"),
    ("3ert", "Oestrogen receptor + tamoxifen", "ER"),
    ("1stp", "Streptavidin + biotin", "Streptavidin"),
    ("2gbp", "Glucose-binding protein", "GBP"),
    ("4hhb", "Haemoglobin + haem", "Haemoglobin"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="best_binding_site_gnn.pt")
    ap.add_argument("--out", default="docs/predictions.json")
    ap.add_argument("--metrics", default="best_metrics.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    proteins = {}

    for pdb_id, name, short in TARGETS:
        path = fetch_pdb(pdb_id)
        g, probs = predict(path, args.checkpoint)
        y = g["y"].numpy()
        coords = g["coords"]

        # Centre and round: the viewer only needs ~0.1 A precision.
        coords = np.round(coords - coords.mean(axis=0), 1)

        entry = {
            "id": pdb_id,
            "name": name,
            "short": short,
            "coords": coords.tolist(),
            "p": [round(float(v), 4) for v in probs],
            "y": [int(v) for v in y],
            "res": [f"{n} {c}{num}" for c, num, n in g["residues"]],
            "edges": int(g["edge_index"].shape[1] // 2),
            "n_pos": int(y.sum()),
            "pr_auc": float(average_precision_score(y, probs)) if 0 < y.sum() < len(y) else 0.0,
            "roc_auc": float(roc_auc_score(y, probs)) if 0 < y.sum() < len(y) else 0.0,
        }
        proteins[pdb_id] = entry
        print(f"{pdb_id}: {len(y)} residues, PR-AUC {entry['pr_auc']:.3f}, "
              f"ROC-AUC {entry['roc_auc']:.3f}")

    val = {}
    if os.path.exists(args.metrics):
        val = json.load(open(args.metrics)).get("best", {})

    honest = (
        f"Validation PR-AUC is <code>{val.get('pr_auc', 0):.3f}</code> against a base rate of "
        f"<code>{val.get('base_rate', 0):.3f}</code> — about "
        f"{val.get('pr_auc', 0) / max(val.get('base_rate', 1e-9), 1e-9):.1f}× better than "
        "guessing, with ROC-AUC "
        f"<code>{val.get('roc_auc', 0):.3f}</code>. That is a useful ranking signal, not a "
        "pocket detector: the model surfaces plausible regions, and the top residues are "
        "enriched for real contacts, but it will also light up hydrophobic patches that bind "
        "nothing. Labels come from whatever ligand was crystallised with each structure, so a "
        "residue marked negative may still be part of a pocket for some other molecule."
    )

    json.dump({"proteins": proteins, "honest": honest}, open(args.out, "w"),
              separators=(",", ":"))
    size = os.path.getsize(args.out) / 1024
    print(f"\nwrote {args.out} ({size:.0f} KB)")


if __name__ == "__main__":
    main()

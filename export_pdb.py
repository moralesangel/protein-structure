"""Write demo PDB files with the predicted probability in the B-factor column.

This is the same trick AlphaFold uses to ship pLDDT: molecular viewers already
know how to colour by B-factor, so the prediction travels inside a standard
file with no custom colour-theme plumbing in the browser.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from predict import fetch_pdb, predict
from secondary import assign as assign_ss

TARGETS = [
    ("1m17", "EGFR kinase + erlotinib", "EGFR"),
    ("3ert", "Oestrogen receptor + tamoxifen", "ER"),
    ("1stp", "Streptavidin + biotin", "Streptavidin"),
    ("2gbp", "Glucose-binding protein", "GBP"),
    ("4hhb", "Haemoglobin + haem", "Haemoglobin"),
]

IGNORED_HETATM = {
    "HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "SO4", "PO4",
    "EDO", "DMS", "ACT", "GOL", "PEG", "MPD",
}


def write_scored_pdb(src_path: str, out_path: str, scores: dict[tuple[str, int], float],
                     keep_ligands: bool = True,
                     drop: set[tuple[str, int]] | None = None) -> int:
    """Copy protein atoms, replacing B-factors with the per-residue score.

    Residues with no score (or ligands) get 0, so the viewer draws them at the
    cold end of the ramp. Ligand HETATM records are kept so the demo can show
    what the model was actually scored against.
    """
    written = 0
    with open(src_path) as fh, open(out_path, "w") as out:
        out.write("REMARK   1 B-FACTOR COLUMN HOLDS PREDICTED BINDING PROBABILITY (0-100)\n")
        for line in fh:
            rec = line[:6]
            if rec == "ENDMDL":
                break  # only the first model
            if rec not in ("ATOM  ", "HETATM"):
                continue
            resname = line[17:20].strip()
            if rec == "HETATM":
                if not keep_ligands or resname in IGNORED_HETATM:
                    continue
                out.write(line[:60] + "  0.00" + line[66:])
                written += 1
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            if drop and (chain, resnum) in drop:
                continue
            score = scores.get((chain, resnum), 0.0)
            out.write(f"{line[:60]}{score * 100:6.2f}{line[66:]}")
            written += 1
        out.write("END\n")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="best_binding_site_gnn.pt")
    ap.add_argument("--outdir", default="docs/structures")
    ap.add_argument("--meta", default="docs/predictions.json")
    ap.add_argument("--metrics", default="best_metrics.json")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    from sklearn.metrics import average_precision_score, roc_auc_score

    proteins = {}
    for pdb_id, name, short in TARGETS:
        path = fetch_pdb(pdb_id)
        g, probs = predict(path, args.checkpoint)
        y = g["y"].numpy()

        scores = {(c, num): float(p) for (c, num, _), p in zip(g["residues"], probs)}

        # Several entries carry a floppy terminal tail far from the body of the
        # structure. Keeping it makes the viewer frame mostly empty space, so
        # residues past 2.2x the median radius are dropped from the demo file.
        centre = g["coords"].mean(axis=0)
        radii = np.linalg.norm(g["coords"] - centre, axis=1)
        limit = float(np.median(radii)) * 2.2
        drop = {
            (c, num) for (c, num, _), r in zip(g["residues"], radii) if r > limit
        }
        if drop:
            print(f"  {pdb_id}: trimming {len(drop)} outlying residues "
                  f"beyond {limit:.0f} A")

        out_pdb = os.path.join(args.outdir, f"{pdb_id}.pdb")
        atoms = write_scored_pdb(path, out_pdb, scores, drop=drop)

        chains = [c for c, _, _ in g["residues"]]
        breaks = {i for i in range(1, len(chains)) if chains[i] != chains[i - 1]}

        order = np.argsort(-probs)[:10]
        proteins[pdb_id] = {
            "id": pdb_id,
            "name": name,
            "short": short,
            "file": f"structures/{pdb_id}.pdb",
            "n_res": len(y),
            "edges": int(g["edge_index"].shape[1] // 2),
            "n_pos": int(y.sum()),
            "pr_auc": float(average_precision_score(y, probs)) if 0 < y.sum() < len(y) else 0.0,
            "roc_auc": float(roc_auc_score(y, probs)) if 0 < y.sum() < len(y) else 0.0,
            "ss": assign_ss(g["coords"], breaks),
            "top": [
                {
                    "label": f"{g['residues'][i][2]} {g['residues'][i][0]}{g['residues'][i][1]}",
                    "chain": g["residues"][i][0],
                    "num": int(g["residues"][i][1]),
                    "p": round(float(probs[i]), 4),
                    "hit": bool(y[i] > 0.5),
                }
                for i in order
            ],
            # Full range, so nothing clamps outside the ramp and renders black.
            # Absolute probabilities sit in a narrow band, so stretching the
            # colours over this range rather than 0-1 is what makes the
            # prediction visible at all.
            "p_min": round(float(probs.min()), 4),
            "p_max": round(float(probs.max()), 4),
        }
        size = os.path.getsize(out_pdb) / 1024
        print(f"{pdb_id}: {atoms} atoms, {size:.0f} KB, PR-AUC {proteins[pdb_id]['pr_auc']:.3f}")

    val = json.load(open(args.metrics)).get("best", {}) if os.path.exists(args.metrics) else {}
    honest = (
        f"Validation PR-AUC is <code>{val.get('pr_auc', 0):.3f}</code> against a base rate of "
        f"<code>{val.get('base_rate', 0):.3f}</code> — about "
        f"{val.get('pr_auc', 0) / max(val.get('base_rate', 1e-9), 1e-9):.1f}× better than "
        f"guessing, with ROC-AUC <code>{val.get('roc_auc', 0):.3f}</code>. That is a useful "
        "ranking signal, not a pocket detector: the model surfaces plausible regions, and the "
        "top residues are enriched for real contacts, but it will also light up hydrophobic "
        "patches that bind nothing. Labels come from whatever ligand was crystallised with each "
        "structure, so a residue marked negative may still be part of a pocket for some other "
        "molecule."
    )

    json.dump({"proteins": proteins, "honest": honest}, open(args.meta, "w"),
              separators=(",", ":"))
    print(f"\nwrote {args.meta}")


if __name__ == "__main__":
    main()

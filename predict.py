"""Binding-site prediction for a single PDB structure.

Reproduces the graph construction from binding_site_gnn.ipynb and runs the
trained checkpoint, so a structure can be scored without opening the notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from Bio.PDB import PDBParser, ShrakeRupley
from scipy.spatial import cKDTree

from features import geometric

AA_MAP = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
NUM_AA_CLASSES = len(AA_MAP) + 1

IGNORED_HETATM = {
    "HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "SO4", "PO4",
    "EDO", "DMS", "ACT", "GOL", "PEG", "MPD",
}

KD_HYDROPHOBICITY = {
    "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
    "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
    "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
}

CHARGE = {"ARG": 1.0, "LYS": 1.0, "HIS": 0.1, "ASP": -1.0, "GLU": -1.0}

MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}


class DistanceAwareConv(nn.Module):
    """Message passing over [x_i, x_j, distance], mean-aggregated.

    Written with dense index_add rather than torch_geometric so the script has
    no PyG dependency; the maths and parameter names match the notebook.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 1):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.SiLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_mlp(torch.cat([x[dst], x[src], edge_attr], dim=-1))
        agg = torch.zeros(x.size(0), msg.size(1), dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        counts = torch.zeros(x.size(0), 1, dtype=msg.dtype)
        counts.index_add_(0, dst, torch.ones(dst.size(0), 1, dtype=msg.dtype))
        agg = agg / counts.clamp(min=1)
        return self.update_mlp(torch.cat([x, agg], dim=-1))


class BindingSiteGNN(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=64, num_layers=2,
                 edge_dim=1, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(DistanceAwareConv(hidden_channels, hidden_channels, edge_dim))
            self.norms.append(nn.LayerNorm(hidden_channels))
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, edge_attr):
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))
        return self.classifier(h).squeeze(-1)


def fetch_pdb(pdb_id: str, dest_dir: str = "./data/raw_pdb") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"{pdb_id.lower()}.pdb")
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        urllib.request.urlretrieve(url, path)
    return path


def build_graph(pdb_path: str, cutoff_radius: float = 8.0,
                contact_threshold: float = 5.0):
    """Residue graph: C-alpha nodes, 8 A radius edges, ligand-contact labels."""
    structure = PDBParser(QUIET=True).get_structure("s", pdb_path)
    ShrakeRupley().compute(structure, level="R")

    model = next(iter(structure))
    residues, coords, sasa = [], [], []
    ligand_atoms = []

    for chain in model:
        for res in chain:
            het, _, _ = res.get_id()
            name = res.get_resname().strip()
            if het == " ":
                if "CA" not in res:
                    continue
                residues.append((chain.get_id(), res.get_id()[1], name))
                coords.append(res["CA"].get_coord())
                sasa.append(res.sasa)
            elif name not in IGNORED_HETATM:
                ligand_atoms.extend(a.get_coord() for a in res)

    if len(residues) < 10:
        raise ValueError("too few residues to build a graph")

    coords = np.asarray(coords, dtype=np.float32)

    feats = np.zeros((len(residues), NUM_AA_CLASSES + 3), dtype=np.float32)
    for i, (_, _, name) in enumerate(residues):
        feats[i, AA_MAP.get(name, NUM_AA_CLASSES - 1)] = 1.0
        feats[i, NUM_AA_CLASSES] = KD_HYDROPHOBICITY.get(name, 0.0) / 4.5
        feats[i, NUM_AA_CLASSES + 1] = CHARGE.get(name, 0.0)
        feats[i, NUM_AA_CLASSES + 2] = min(sasa[i] / MAX_ASA.get(name, 200), 1.0)

    # Pocket geometry: chemistry alone does not say where a hollow is.
    feats = np.concatenate([feats, geometric(coords, cutoff_radius)], axis=1)

    tree = cKDTree(coords)
    pairs = tree.query_pairs(cutoff_radius, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("no contacts within the cutoff")
    edge_index = np.concatenate([pairs.T, pairs.T[::-1]], axis=1)
    d = np.linalg.norm(coords[edge_index[0]] - coords[edge_index[1]], axis=1)
    edge_attr = (d / cutoff_radius).astype(np.float32)[:, None]

    labels = np.zeros(len(residues), dtype=np.float32)
    if ligand_atoms:
        lig = np.asarray(ligand_atoms, dtype=np.float32)
        close = cKDTree(lig).query_ball_point(coords, contact_threshold)
        labels = np.array([1.0 if c else 0.0 for c in close], dtype=np.float32)

    return {
        "x": torch.from_numpy(feats),
        "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
        "edge_attr": torch.from_numpy(edge_attr),
        "y": torch.from_numpy(labels),
        "coords": coords,
        "residues": residues,
        "has_ligand": bool(ligand_atoms),
    }


def load_model(checkpoint: str = "best_binding_site_gnn.pt") -> BindingSiteGNN:
    """Rebuild the model with the shape the checkpoint was saved with."""
    state = torch.load(checkpoint, map_location="cpu")
    in_channels = state["input_proj.weight"].shape[1]
    hidden = state["input_proj.weight"].shape[0]
    num_layers = len({k.split(".")[1] for k in state if k.startswith("convs.")})
    model = BindingSiteGNN(in_channels=in_channels, hidden_channels=hidden,
                           num_layers=num_layers)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(pdb_path: str, checkpoint: str = "best_binding_site_gnn.pt"):
    g = build_graph(pdb_path)
    model = load_model(checkpoint)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(g["x"], g["edge_index"], g["edge_attr"]))
    return g, probs.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdb", help="PDB id (downloaded from RCSB) or a local .pdb path")
    ap.add_argument("--checkpoint", default="best_binding_site_gnn.pt")
    ap.add_argument("--json", help="write predictions to this file")
    args = ap.parse_args()

    path = args.pdb if os.path.exists(args.pdb) else fetch_pdb(args.pdb)
    g, probs = predict(path, args.checkpoint)

    order = np.argsort(-probs)
    print(f"{len(g['residues'])} residues, {g['edge_index'].shape[1] // 2} contacts")
    print("\ntop 15 predicted binding-site residues:")
    for i in order[:15]:
        chain, num, name = g["residues"][i]
        mark = ""
        if g["has_ligand"]:
            mark = "  <- contacts ligand" if g["y"][i] > 0.5 else ""
        print(f"  {name} {chain}{num:<5} p={probs[i]:.3f}{mark}")

    if g["has_ligand"]:
        from sklearn.metrics import average_precision_score, roc_auc_score
        y = g["y"].numpy()
        if 0 < y.sum() < len(y):
            print(f"\nthis structure: PR-AUC {average_precision_score(y, probs):.3f} | "
                  f"ROC-AUC {roc_auc_score(y, probs):.3f} | "
                  f"{int(y.sum())} of {len(y)} residues contact the ligand")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "residues": [{"chain": c, "num": int(n), "name": nm,
                              "p": float(p), "label": float(l)}
                             for (c, n, nm), p, l in zip(g["residues"], probs, g["y"])],
                "coords": g["coords"].tolist(),
            }, fh)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

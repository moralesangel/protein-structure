"""Build the residue-graph dataset from downloaded PDB files."""
import sys, glob, os, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
from predict import build_graph
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(ROOT, "data", "graphs.pkl")
paths = sorted(glob.glob(os.path.join(ROOT, "data", "raw_pdb", "*.pdb")))
graphs = []
MAX_BYTES = 8 * 1024 * 1024  # SASA is O(n^2); one 110MB entry stalls the run

for i, p in enumerate(paths):
    if os.path.getsize(p) > MAX_BYTES:
        print(f"skip {os.path.basename(p)} ({os.path.getsize(p)//1024//1024}MB)", flush=True)
        continue
    try:
        g = build_graph(p)
        if g["has_ligand"] and 0 < float(g["y"].sum()) < len(g["y"]):
            graphs.append({k: g[k] for k in ("x","edge_index","edge_attr","y")} |
                          {"id": os.path.basename(p)[:4]})
    except Exception:
        pass
    if (i+1) % 100 == 0:
        print(f"{i+1}/{len(paths)} -> {len(graphs)} usable", flush=True)
pickle.dump(graphs, open(out, "wb"))
n = sum(len(g["y"]) for g in graphs); pos = sum(float(g["y"].sum()) for g in graphs)
print(f"saved {len(graphs)} graphs, {n} residues, positive rate {pos/n:.4f}")

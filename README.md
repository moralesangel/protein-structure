# Binding Site GNN

A graph neural network that scores every residue of a protein for how likely it
is to contact a ligand, trained on structures from the PDB.

**[▶ Try the live demo](https://moralesangel.github.io/protein-structure/)** — rotate
real structures in 3D with every residue coloured by its predicted probability.

---

## The idea

A protein is a natural graph: residues are nodes, and two residues that sit close
in space interact whether or not they are adjacent in the sequence. Binding
pockets are a *geometric* property — a cleft lined by the right chemistry — so a
model that passes messages along spatial neighbours has the right inductive bias,
where a sequence model would have to learn the folding first.

Each structure becomes one graph:

| | |
| :--- | :--- |
| **Nodes** | One per residue, positioned at its Cα |
| **Node features** | Residue identity (21-way one-hot), Kyte–Doolittle hydrophobicity, formal charge, relative SASA |
| **Edges** | Every residue pair within 8 Å |
| **Edge features** | The Cα–Cα distance, normalised by the cutoff |
| **Label** | 1 if the residue has any atom within 5 Å of a crystallised ligand |

Relative SASA matters more than it looks: a residue buried in the core cannot
bind anything, so exposure carries much of the signal about *where* a pocket
can be at all.

## The model

`DistanceAwareConv` is a message-passing layer in the EGNN spirit, without
updating coordinates. For each edge it builds a message from
`[x_i, x_j, distance]` through an MLP, mean-aggregates over neighbours, then
updates the node from `[x_i, aggregated]`:

```python
msg = msg_mlp(cat([x_i, x_j, edge_attr]))        # distance enters explicitly
agg = scatter_mean(msg, dst)
out = update_mlp(cat([x, agg]))
```

Three such layers with residual connections and LayerNorm, then a small
classifier head. LayerNorm rather than BatchNorm because graphs vary enormously
in size and batch statistics are meaningless across them.

## Training and the metric that matters

Only about 7% of residues contact a ligand, so **accuracy is a useless metric** —
a model that answers "never" scores 93%. Two choices follow from that:

- **Focal loss** (α 0.25, γ 2.0) down-weights the easy negatives that would
  otherwise dominate the gradient.
- **Checkpoint selection on validation PR-AUC**, not accuracy or even ROC-AUC.
  Precision–recall is the honest view when positives are rare.

See [`results.md`](results.md) for the numbers and what they do and do not mean.

## Running it

```bash
pip install -r requirements.txt

# Score any PDB entry (downloads it from RCSB)
python predict.py 1M17

# Or a local file
python predict.py path/to/structure.pdb --json out.json
```

Training from scratch needs the structures, which are not committed:

```bash
python scripts/build_graphs.py     # PDB files -> residue graphs
python train.py --epochs 60
python export_demo.py              # writes docs/predictions.json
```

## Files

| File | Purpose |
| :--- | :--- |
| [`predict.py`](predict.py) | Graph construction, the model, and single-structure inference |
| [`train.py`](train.py) | Focal-loss training loop with PR-AUC checkpointing |
| [`export_demo.py`](export_demo.py) | Scores the demo structures into `docs/predictions.json` |
| [`scripts/build_graphs.py`](scripts/build_graphs.py) | Builds the graph dataset from downloaded PDBs |
| [`binding_site_gnn.ipynb`](binding_site_gnn.ipynb) | The original exploration: RCSB fetching, EDA, first training runs |
| [`docs/`](docs/) | The 3D demo published on GitHub Pages |

`predict.py` implements the message passing with plain `index_add_` rather than
`torch_geometric`, so inference needs only PyTorch, Biopython and SciPy.

# Results

Trained on 591 residue graphs from the PDB (141,161 residues), split 85/15.
Best checkpoint at epoch 87, selected on validation PR-AUC.

## Validation

| Metric | Value |
| :--- | ---: |
| PR-AUC | **0.611** |
| ROC-AUC | 0.901 |
| Best F1 | 0.563 (threshold 0.28) |
| Base rate | 0.066 |
| Lift over base rate | 9.3x |

## What that means in practice

Rank every residue in a held-out structure by predicted probability:

| Slice | Precision | Recall | vs random |
| :--- | ---: | ---: | ---: |
| Top 5% | 65.2% | 49.4% | 9.9x |
| Top 10% | 43.1% | 65.3% | 6.5x |
| Top 20% | 26.6% | 80.8% | 4.0x |

Two thirds of the top 5% really do contact a ligand, against 6.6% if you picked
at random, and that slice already recovers half of every binding residue.

## Per-structure, on the demo set

| PDB | Structure | PR-AUC | ROC-AUC | Top 5% precision |
| :--- | :--- | ---: | ---: | ---: |
| 1M17 | EGFR kinase + erlotinib | 0.438 | 0.894 | 27% (base 4.2%) |
| 3ERT | Oestrogen receptor + tamoxifen | 0.511 | 0.946 | 42% (base 3.2%) |
| 1STP | Streptavidin + biotin | 0.891 | 0.990 | 67% (base 4.1%) |
| 2GBP | Glucose-binding protein | 0.181 | 0.969 | 20% (base 1.0%) |
| 4HHB | Haemoglobin + haem | 0.533 | 0.899 | 50% (base 4.4%) |

## What made the difference

The first model used residue chemistry only — identity, hydrophobicity, charge
and SASA — and reached PR-AUC 0.151. Adding six geometric features per residue
took it to 0.611, a four-fold improvement:

| Feature | AUC alone |
| :--- | ---: |
| concavity (does the residue face into a hollow) | 0.632 |
| radial position (how far from the protein centre) | 0.325 (inverse, so 0.675) |
| mid-shell density | 0.558 |
| protrusion, planarity, contact density | ~0.51 each |

Chemistry says what a residue *is*; a pocket is about where it *sits*. Mean
aggregation over neighbours cannot recover concavity on its own, so stating it
explicitly is what unlocked the model.

## Honest limits

- **Labels are ligand-specific.** A residue is positive only if it contacts the
  ligand crystallised in that entry. Allosteric sites, cryptic pockets and
  second sites are all labelled negative, so some "false positives" are real
  pockets for other molecules.
- **Features stop at the Calpha.** Side-chain geometry, which decides whether a
  molecule physically fits, is invisible to the model.
- **2GBP scores poorly** (PR-AUC 0.181) because only 3 of its 309 residues are
  labelled positive; with so few, the metric is unstable even though ROC-AUC is
  0.969.

## What would likely help next

1. All-atom or side-chain-level nodes.
2. Evolutionary features (PSSM or an ESM embedding) — conservation is one of the
   strongest signals for functional sites and is absent here entirely.
3. Predicting pockets as clusters rather than scoring residues independently.

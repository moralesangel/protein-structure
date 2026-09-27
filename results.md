# Results

Trained on 591 residue graphs from the PDB (141,161 residues), split 85/15.
Best checkpoint at epoch 80, selected on validation PR-AUC.

## Validation

| Metric | Value |
| :--- | ---: |
| PR-AUC | **0.151** |
| ROC-AUC | 0.688 |
| Best F1 | 0.214 (at threshold 0.24) |
| Base rate (positives) | 0.066 |
| Lift over base rate | 2.3x |

PR-AUC is the headline because positives are rare: only 6.6% of residues contact a ligand, so a model that always answers "no" is already 93.4% accurate and completely useless.

## Per-structure, on the demo set

| PDB | Structure | PR-AUC | ROC-AUC | Contacts / residues |
| :--- | :--- | ---: | ---: | ---: |
| 1M17 | EGFR kinase + erlotinib | 0.046 | 0.510 | 13 / 312 |
| 3ERT | Oestrogen receptor + tamoxifen | 0.081 | 0.601 | 8 / 246 |
| 1STP | Streptavidin + biotin | 0.610 | 0.919 | 5 / 121 |
| 2GBP | Glucose-binding protein | 0.040 | 0.705 | 3 / 309 |
| 4HHB | Haemoglobin + haem | 0.119 | 0.743 | 25 / 574 |

## What this means

The spread across structures is the honest headline. Streptavidin, with a small
deep biotin pocket, is predicted well (PR-AUC 0.61, ROC-AUC 0.92). EGFR is close
to chance on this metric. The model has learned something real about where
pockets tend to sit -- exposed, hydrophobic, geometrically clustered -- but it
has not learned to identify *a specific* binding site.

Two limits are worth stating plainly:

- **Labels are ligand-specific.** A residue counts as positive only if it
  contacts the ligand that happened to be crystallised in that entry. Allosteric
  sites, cryptic pockets and second binding sites are all labelled negative, so
  some "false positives" may not be false at all.
- **The features are coarse.** Residue identity, hydrophobicity, charge and SASA
  at the Calpha level throw away side-chain geometry, which is much of what
  determines whether a pocket can actually accommodate a molecule.

## What would likely help

1. All-atom or side-chain-level nodes instead of one node per residue.
2. Evolutionary features (a PSSM or an ESM embedding) -- conservation is one of
   the strongest signals for functional sites and is absent here entirely.
3. Predicting pockets as clusters rather than scoring residues independently,
   which is closer to what the task actually is.

"""Train the binding-site GNN on the prebuilt residue graphs.

Uses focal loss for the class imbalance (~7% of residues contact a ligand) and
selects the checkpoint by validation PR-AUC, which is the metric that matters
when positives are rare.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from predict import BindingSiteGNN


class FocalLoss(nn.Module):
    """Binary focal loss: down-weights the easy negatives that dominate."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def evaluate(model, graphs):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for g in graphs:
            logits = model(g["x"], g["edge_index"], g["edge_attr"])
            ps.append(torch.sigmoid(logits).numpy())
            ys.append(g["y"].numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)

    pr = average_precision_score(y, p)
    roc = roc_auc_score(y, p)

    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        pred = p >= thr
        tp = float((pred & (y > 0.5)).sum())
        if tp == 0:
            continue
        prec = tp / pred.sum()
        rec = tp / (y > 0.5).sum()
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    return {"pr_auc": pr, "roc_auc": roc, "best_f1": best_f1, "best_threshold": best_thr,
            "base_rate": float((y > 0.5).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", default="data/graphs.pkl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out", default="best_binding_site_gnn.pt")
    ap.add_argument("--history", default="training_history.json")
    args = ap.parse_args()

    graphs = pickle.load(open(args.graphs, "rb"))
    random.Random(42).shuffle(graphs)
    split = int(len(graphs) * 0.85)
    train_set, val_set = graphs[:split], graphs[split:]
    print(f"{len(train_set)} train / {len(val_set)} val graphs")

    torch.manual_seed(42)
    model = BindingSiteGNN(in_channels=train_set[0]["x"].shape[1],
                           hidden_channels=args.hidden, num_layers=args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=4)
    crit = FocalLoss()

    best, bad, history = 0.0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_set)
        total = 0.0
        opt.zero_grad()
        for i, g in enumerate(train_set, 1):
            loss = crit(model(g["x"], g["edge_index"], g["edge_attr"]), g["y"]) / args.batch
            loss.backward()
            total += loss.item() * args.batch
            if i % args.batch == 0 or i == len(train_set):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

        m = evaluate(model, val_set)
        sched.step(m["pr_auc"])
        history.append({"epoch": epoch, "loss": total / len(train_set), **m})
        print(f"epoch {epoch:03d} | loss {total/len(train_set):.4f} | "
              f"PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | F1 {m['best_f1']:.4f}")

        if m["pr_auc"] > best:
            best, bad = m["pr_auc"], 0
            torch.save(model.state_dict(), args.out)
            json.dump({"best": m, "epoch": epoch}, open("best_metrics.json", "w"), indent=1)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop at epoch {epoch}")
                break

    json.dump(history, open(args.history, "w"), indent=1)
    print(f"\nbest validation PR-AUC {best:.4f} (base rate {history[-1]['base_rate']:.4f})")


if __name__ == "__main__":
    main()

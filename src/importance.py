"""Per-fold gain and permutation importance (held-out fold, top-3 drop) for the binary reference model.

  python -m src.importance
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features, models
from src.cv import controller_folds, fold_array
from src.data_io import ROOT, load_all, load_config
from src.models import Groups, top3_hits


def main() -> None:
    cfg = load_config()
    tr = load_all(cfg)["train"]
    trf, feats = features.build(tr)
    folds = fold_array(tr, controller_folds(tr, cfg["n_folds"]))
    R = int(np.load(ROOT / "cache" / "preds" / "lgb_binary.npz")["R"])
    rng = np.random.default_rng(cfg["seed"])
    gains, perms = {}, {}
    for k in range(cfg["n_folds"]):
        b, _ = models.fit(trf[folds != k], feats, dict(objective="binary"), cfg["lgb_base"], R, seed=cfg["seed"])
        g = pd.Series(b.feature_importance("gain"), index=feats)
        gains[k] = g / g.sum()
        ev = trf[folds == k].copy()
        grp = Groups.of(ev)
        base = top3_hits(models.predict(b, ev, feats), grp).mean()
        top = g.sort_values(ascending=False).index[:20]
        sid = ev["sample_id"].to_numpy()
        drops = {}
        for f in top:  # permute across samples (whole-column shuffle) on the held-out fold
            x = ev[f].to_numpy().copy()
            ev[f] = x[rng.permutation(len(x))]
            drops[f] = base - top3_hits(models.predict(b, ev, feats), grp).mean()
            ev[f] = x
        perms[k] = pd.Series(drops)
    G, P = pd.DataFrame(gains), pd.DataFrame(perms)
    G["mean"], G["rank_std"] = G.mean(1), G.rank(ascending=False).std(1)
    P["mean"] = P.mean(1)
    out = ["## Importance stability across CV folds (binary reference, seed 42, R=%d)" % R, "",
           "Gain share per fold (top 20 by mean):", "", G.sort_values("mean", ascending=False).head(20).round(4).to_markdown(),
           "", "Permutation importance: drop in held-out top-3 hit rate when the column is shuffled (top-20 gain features):",
           "", P.sort_values("mean", ascending=False).round(4).to_markdown(), ""]
    (ROOT / "reports" / "importance.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()

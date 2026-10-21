"""Internal metric must equal scorer.py exactly.  python -m tests.test_metric_vs_scorer"""
import subprocess
import sys

import numpy as np

from src import baselines
from src.data_io import ROOT, load_all
from src.metrics import hits_from_scores, hits_from_submission, labels_of
from src.models import Groups, top3_hits
from src.rank import topk, write_submission


def scorer_value(path) -> float:
    out = subprocess.run([sys.executable, str(ROOT / "scorer.py"), "--labelled", str(ROOT / "validation.csv"),
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.split("top-3 hit rate:")[1].split()[0])


def main():
    d = load_all()
    tr, va = d["train"], d["validation"]
    tmp = ROOT / "cache" / "tmp_metric_check.csv"
    for name, sc in (("rule", baselines.rule_score(tr, va)), ("random", baselines.random_score(va, 0)),
                     ("ties", np.zeros(len(va)))):
        top = topk(va, sc)
        sub = write_submission(top, tmp)
        internal = hits_from_scores(va, sc).mean()
        via_sub = hits_from_submission(sub, labels_of(va)).mean()
        fast = top3_hits(sc.astype(float), Groups.of(va)).mean()
        ext = scorer_value(tmp)
        print(f"{name}: internal {internal:.6f} parsed-sub {via_sub:.6f} fast {fast:.6f} scorer.py {ext:.4f}")
        assert abs(internal - via_sub) < 1e-12 and abs(internal - fast) < 1e-12
        assert round(internal, 4) == ext, "internal metric disagrees with scorer.py"
    tmp.unlink()
    print("metric matches scorer.py")


if __name__ == "__main__":
    main()

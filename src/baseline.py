"""Baseline degli organizzatori (M1): ordina per |time_to_sector| crescente."""
import subprocess
import sys

from src.config import ROOT_DIR, SUB_DIR
from src.data import load
from src.metric import top3_hit_rate, to_submission


def baseline_score(df):
    # -|tts| decrescente; NaN in coda
    return -df["time_to_sector"].abs().fillna(float("inf"))


if __name__ == "__main__":
    val = load("validation")
    val["score"] = baseline_score(val)
    out = SUB_DIR / "baseline_validation.csv"
    to_submission(val, "score").to_csv(out, index=False)
    print(f"metric.py : {top3_hit_rate(val, 'score'):.4f}")
    subprocess.run([sys.executable, str(ROOT_DIR / "scorer.py"), "--labelled",
                    str(ROOT_DIR / "data" / "validation.csv"), str(out)], check=True)

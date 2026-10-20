#!/usr/bin/env bash
# Esegue un esperimento su qualsiasi PC (Linux/macOS, o Windows con WSL/Git Bash).
# Uso:  ./run.sh <modello> <feature_set> <exp_id>        es. ./run.sh cat_rank ac m4-cat-ac
# Modelli: lgbm_rank | lgbm_xendcg | cat_rank.  Variabili: AWARE_THREADS=<n>, AWARE_GPU=1 (solo CatBoost).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=${1:-cat_rank}; FEATURES=${2:-ac}; EXP_ID=${3:-${MODEL}-${FEATURES}}

PY=python3; command -v $PY >/dev/null || PY=python
if [ ! -d .venv ]; then $PY -m venv .venv; fi
VPY=.venv/bin/python; [ -x "$VPY" ] || VPY=.venv/Scripts/python.exe   # Windows
$VPY -m pip install -q -r requirements.txt

for f in train validation test; do
  [ -f "data/$f.csv" ] || { echo "manca data/$f.csv: copia i CSV della competizione in data/"; exit 1; }
done
[ -f data/train.parquet ] || $VPY -m src.data --convert
$VPY -m pytest -q tests
echo "== CV $MODEL / $FEATURES su $($VPY -c 'from src.config import N_THREADS; print(N_THREADS)') thread"
$VPY -m src.cv --model "$MODEL" --features "$FEATURES" --exp-id "$EXP_ID" --milestone M4 --note "run.sh su $(hostname)"
echo "== fatto. File da riportare: experiments/oof/$EXP_ID.parquet, experiments/oof/$EXP_ID.json, ultima riga di experiments/EXPERIMENTS.md"

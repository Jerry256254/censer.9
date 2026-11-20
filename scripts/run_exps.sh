#!/bin/bash
# Run training experiments sequentially, then evaluate each.
cd /home/tvadmin/censer.5
export PYTHONPATH=/home/tvadmin/censer.5
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
PY=/home/tvadmin/censer/.venv/bin/python
for name in "$@"; do
  echo "================ EXP $name ================"
  $PY -m kuclab.train --config configs/exp/$name.json --data data/processed --out runs/$name 2>&1 | tail -25
  $PY scripts/eval_ckpt.py --checkpoint runs/$name/best.pt
done
echo ALL_DONE

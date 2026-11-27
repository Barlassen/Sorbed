#!/bin/sh
# Staging step 3, B arms (HANDOFF 2026-11-04): B(a) petchs only, B(b) petchs + fr7kn extra sources; 3 folds x 3 seeds.
set -e
for arm in ba bb; do
  flag=""
  [ "$arm" = bb ] && flag="--extra"
  for seed in 0 1 2; do
    for k in 0 1 2; do
      out=runs/staging/step3/${arm}_fold${k}_s${seed}
      [ -f "$out.pt" ] && continue
      PYTHONPATH=. .venv/bin/python -m training.train_stage_cnn --fold $k --init coco --seed $seed $flag \
        --out "$out.json" --save "$out.pt" > logs/step3_${arm}_fold${k}_s${seed}.log 2>&1
    done
  done
done

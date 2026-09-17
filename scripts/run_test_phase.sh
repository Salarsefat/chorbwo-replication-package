#!/bin/bash
# Held-out test phase with the frozen design. Run sequentially (single core).
cd "$(dirname "$0")/.."
python3 scripts/run_experiments.py --phase test --group main     --out results/test_main.jsonl     >> results/test_main.log 2>&1
python3 scripts/run_experiments.py --phase test --group optonly  --out results/test_optonly.jsonl  >> results/test_optonly.log 2>&1
python3 scripts/run_experiments.py --phase test --group ablation --out results/test_ablation.jsonl >> results/test_ablation.log 2>&1
echo "TEST PHASE DONE $(date)" >> results/test_main.log

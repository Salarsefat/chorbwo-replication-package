#!/bin/bash
# Restart-safe driver: (re)launches the test-phase steps until all are done (runner is resume-safe).
cd "$(dirname "$0")/.."
run() { python3 scripts/run_experiments.py "$@" >> results/watchdog.log 2>&1; }
until [ -f results/.main_done ]; do run --phase test --group main --out results/test_main.jsonl && touch results/.main_done; done
until [ -f results/.optonly_done ]; do run --phase test --group optonly --out results/test_optonly.jsonl && touch results/.optonly_done; done
until [ -f results/.ablation_done ]; do run --phase test --group ablation --out results/test_ablation.jsonl && touch results/.ablation_done; done
for sw in 0 200 1000; do
  until [ -f results/.sens${sw}_done ]; do run --phase test --group sens --switch-ms $sw --configs synthetic:10:20,wsdream:10:20 --out results/test_sens_sw${sw}.jsonl && touch results/.sens${sw}_done; done
done
until [ -f results/.regime_done ]; do run --phase test --group main --configs synthetic-regime:10:20,synthetic-regime:20:20 --out results/test_regime.jsonl && touch results/.regime_done; done
echo "ALL DONE $(date)" >> results/watchdog.log

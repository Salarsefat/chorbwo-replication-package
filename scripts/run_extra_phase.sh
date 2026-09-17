#!/bin/bash
# Side studies (run after the test phase): switching-cost sensitivity and model misspecification.
cd "$(dirname "$0")/.."
while pgrep -f run_test_phase.sh > /dev/null; do sleep 30; done
for sw in 0 200 1000; do
  python3 scripts/run_experiments.py --phase test --group sens --switch-ms $sw \
      --configs synthetic:10:20,wsdream:10:20 --out results/test_sens_sw${sw}.jsonl >> results/test_sens.log 2>&1
done
python3 scripts/run_experiments.py --phase test --group main \
    --configs synthetic-regime:10:20,synthetic-regime:20:20 --out results/test_regime.jsonl >> results/test_regime.log 2>&1
echo "EXTRA PHASE DONE $(date)" >> results/test_sens.log

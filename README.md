# chorbwo: MDP-BWO for QoS-aware service choreography in the IoT

Replication package for the manuscript

> S. S. Sefati, A. Vlahov, R. Craciunescu, B. Arasteh, V. Poulkov,
> **Service Choreography in the Internet of Things Based on Forecast-Guided Binding and Local Rebinding**,
> manuscript under review, 2026. *(Journal, volume and DOI will be added on publication.)*

MDP-BWO learns a three-state Markov model (nominal / degraded / faulty) of every candidate service
from its QoS history. The model drives two things:

- a forecast-guided **Black Widow Optimization** search that binds each choreography role to a
  service before execution;
- a per-participant **finite-horizon MDP** rebinding policy, solved offline and executed at run time
  as a table lookup.

The repository contains the simulator, the instance generators, the six baselines, the frozen
experimental protocol, the raw run records and the scripts that regenerate every table and figure
of the paper.

## Repository layout

```
chorbwo/            Python package
  instances.py        instance generators (synthetic, WS-DREAM, synthetic with a hidden regime)
  markov.py           learning the per-service DTMC; static / last-observed / Markov QoS estimators
  evaluate.py         vectorized evaluation of bindings (aggregation, utility, constraints)
  optimizers.py       MDP-BWO, BWO, GA, PSO, GWO, ACO, MILP, greedy
  runtime.py          shortlists, MDP rebinding policies, execution simulator
  experiment.py       one case = build instance, learn models, optimize, execute
configs/
  protocol.yaml       budget, sessions, horizon, seeds and configurations of each phase
  methods.yaml        method specifications of every experiment group
  tuned_params_frozen.json   hyperparameters used for all test runs (frozen, see FROZEN_AT.txt)
  tuned_params.json   output of scripts/tune.py (identical to the frozen file as shipped)
scripts/            runners, analysis and checks (see below)
data/               WS-DREAM dataset #1 (third-party, MIT; see data/LICENSE)
results/            raw run records (*.jsonl.gz), tuning log, summary CSVs
paper/tables/       LaTeX tables of the paper (generated)
paper/figures/      vector figures of the paper (generated)
```

## Installation

Python 3.12 is used in CI. Get the code by cloning the repository or by unpacking the archive
you received, then work from the `chorbwo` folder:

```bash
git clone https://github.com/Salarsefat/chorbwo.git   # or unpack the archive
cd chorbwo
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The scripts put the repository root on the Python path, so no package installation is needed.
`requirements.txt` pins the versions with which the tables and figures were verified to regenerate
exactly.

## Quick start

```bash
# one configuration, two seeds, all seven methods of the main comparison (about 10 s)
python scripts/run_experiments.py --phase test --group main \
    --configs synthetic:10:20 --seeds 1000:1002 --out /tmp/smoke.jsonl
python scripts/summarize.py /tmp/smoke.jsonl
```

## Regenerating the paper's tables and figures from the shipped records

```bash
python scripts/analyze.py all      # tables, summary CSVs, forecast study (about 25 s)
python scripts/figures_pro.py      # publication figures (about 30 s; needs analyze.py first)
python scripts/export_prism.py     # optional: figure data for GraphPad Prism -> paper/prism/
```

`analyze.py` reads `results/<name>.jsonl` if present, otherwise the shipped
`results/<name>.jsonl.gz`. It accepts `main`, `ablation`, `sens`, `regime` or `forecast` instead of
`all`. `figures_pro.py` also writes 600-dpi PNG and TIFF copies; these are not committed. The
`.pzfx` files written by `export_prism.py` have not been validated in GraphPad Prism.

| Study | Records | Tables (`paper/tables/`) | Figures (`paper/figures/`) |
|---|---|---|---|
| Main comparison: 7 methods × 14 configurations × 30 seeds | `test_main` | `main_results`, `stats_summary`, `metrics_pooled_syn`, `metrics_pooled_wsd`, `friedman_ranks`, `metric_directions`, `timing`, `milp_status` | `main_U`, `main_solved`, `main_rel`, `main_thr`, `main_lat`, `main_cost`, `main_eng`, `main_t_opt` |
| Optimizer-only comparison (every optimizer gets the Markov forecast and the reactive runtime) | `test_optonly` (+ `test_main`) | `optonly` | `convergence` |
| Ablation A0–A13 | `test_ablation` | `ablation` | `ablation` |
| Handshake-latency sensitivity (0, 200, 1000 ms; m = 10, n = 20; no MILP) | `test_sens_sw0`, `_sw200`, `_sw1000`; the 40 ms point comes from `test_main`, and for MDP-BWO+R0 from ablation variant A10 | — | `sensitivity_switch` |
| Hidden global regime, i.e. model misspecification (synthetic; (10, 20) and (20, 20)) | `test_regime` | `regime` | — |
| Forecast accuracy (recomputed: seeds 1000–1009, m = 10, n = 20, 100 sessions) | — | `forecast` | `forecast` |
| Hyperparameter tuning | `tuning.jsonl.gz`, `tune.log` | — | — |
| Design iteration, never reported (`dev_ablation_v0` uses an earlier variant numbering) | `dev_main_v0`, `dev_ablation_v0` | — | — |

`main_results` reports mean (SD) over seeds; the other tables report means.

- **Significance marks** in `main_results`, `regime` and `optonly` come from two-sided Wilcoxon
  signed-rank tests, paired over seeds and Holm-corrected within a configuration. The reference is
  MDP-BWO, or MDP-BWO+R0 in `optonly`.
- **`ablation`** compares each variant with A0 over all configurations of a dataset, Holm-corrected
  within the dataset.
- **Figure error bars** are bootstrap 95 % confidence intervals; the convergence plot shades ±1 SD.

## Re-running the experiments

| Step | Command | Recorded cost |
|---|---|---|
| Tuning on the validation seeds | `python scripts/tune.py \| tee results/tune.log` | about 2 min of trial time |
| Test phase: main, optimizer-only, ablation | `bash scripts/run_test_phase.sh` | about 40 min of optimizer time |
| Side studies: sensitivity, regime | `bash scripts/run_extra_phase.sh` | about 8 min of optimizer time |
| Restart-safe driver for all test steps | `bash scripts/watchdog.sh` | — |

The recorded costs are optimizer and policy-construction time logged in the records (one process).
Wall-clock time is somewhat longer and depends on the machine; more than half of the test-phase
time is the MILP baseline.

**Resuming.** The runners append to `results/<name>.jsonl` and skip cases already present, so an
interrupted run resumes where it stopped. They do not read the shipped `.jsonl.gz` files. To
replace the shipped records, run the steps above, then compress the new files over the old ones:
`gzip -kf results/test_*.jsonl`, plus `results/tuning.jsonl` if you re-tuned.

**Hyperparameters.** `tune.py` writes `configs/tuned_params.json`. The test runs read
`configs/tuned_params_frozen.json`, so re-tuning never changes them silently. To adopt new values,
copy the file over the frozen one and update `configs/FROZEN_AT.txt`.

**Platform.** The shell scripts need bash; `run_extra_phase.sh` also uses `pgrep`. On Windows, use
WSL or Git Bash, or call `scripts/run_experiments.py` directly with the arguments in the scripts.

### Protocol (`configs/protocol.yaml`)

| Setting | Value |
|---|---|
| Fitness evaluations per optimizer run | 3000 |
| Executed sessions per binding (E) × decision slots per session (H) | 30 × 20 |
| Alternatives per shortlist (K) | 2 |
| MILP time limit | 10 s (HiGHS via SciPy) |
| Seeds | dev 0–4, validation 100–104, test 1000–1029 |
| Test configurations (m roles, n candidates per role) | (10, 10), (10, 20), (10, 30), (10, 50), (5, 20), (15, 20), (20, 20) on both datasets |

## Method summary

**Instances.** An instance has m roles connected by a random layered message-flow DAG, with n
candidate services per role. Each candidate has:

- a three-state DTMC with state-dependent latency, reliability, throughput and energy, plus a
  static cost;
- peer-to-peer link latencies to the candidates of connected roles;
- a 400-slot historical trace, from which the planner learns its model;
- an observed state at session start.

**Evaluation and constraints.**

- End-to-end latency is the critical path including links. Cost and energy are sums (energy
  includes link energy), reliability is a product, and throughput is the minimum over roles.
- The utility is a weighted sum of normalized QoS. Relative constraint violations are penalized
  with weight 2.
- Latency, reliability, throughput (and cost, for synthetic data) thresholds are the medians over
  2,000 random bindings, evaluated with the `static` estimator.

**Estimators.**

- `static`: the trace average.
- `last`: the QoS of the observed state, assumed to persist.
- `markov`: the expected session QoS under the learned DTMC, starting from the observed state and
  discounted with γ (tuned γ = 0.5).

**Planner (MDP-BWO).** A discrete Black Widow Optimization with three modifications, each switchable
for the ablation:

- M1 seeds part of the initial population from a softmax over per-candidate scores;
- M2 re-samples mutated roles from the same softmax;
- M3 adapts the cannibalism rate and ranks by feasibility first.

**Runtime.**

- *MDP-BWO:* each role keeps its planned service plus K alternatives. The alternatives are chosen
  from the six best one-role-lookahead candidates so as to maximize the value of a finite-horizon
  MDP. The MDP is solved by value iteration with the learned DTMCs and charges the handshake
  latency of each rebinding.
- *Baselines:* reactive fail-over, i.e. switch to the best-ranked non-faulty alternative only when
  the bound service is faulty.

**Baselines.**

- BWO, GA, PSO, GWO, ACO (MAX–MIN ant system).
- MILP on a static linearized surrogate.

Each baseline uses whichever of the `static` and `last` estimators scored better on the validation
seeds.

**Execution.** E sessions of H slots are simulated with the true (hidden) state trajectories,
lognormal latency noise (CV 0.25) and random failures. All methods of a case share the same
execution random numbers.

## Data

- **Synthetic** instances are generated from the seed alone.
- **WS-DREAM dataset #1** (`data/`) holds response time and throughput from 339 users on 5,825 web
  services, plus user and service lists (see `data/readme.txt`).
  - The generator keeps the 4,761 services that have a valid response time from at least 200 users
    and known coordinates.
  - Latency levels are the 25th/75th/95th percentiles of each service's response time, and
    throughput levels the 75th/25th/5th percentiles.
  - Nominal reliability is the fraction of users with a valid measurement, clipped to [0.60, 0.999].
  - State dynamics are synthetic, and link latency grows with the great-circle distance between
    services.
  - Cost and energy are not available, so their weights are 0; the weights are latency 0.40,
    reliability 0.35 and throughput 0.25. The synthetic weights are 0.30, 0.25, 0.15, 0.15
    and 0.15 for latency, reliability, throughput, cost and energy.
  - `userlist.txt` is not read by the code; it is kept as part of the dataset.

The WS-DREAM files are distributed under the WS-DREAM project's MIT License (`data/LICENSE`). If
you use them, cite the WS-DREAM paper below.

## Record format (`results/*.jsonl.gz`)

One JSON object per (dataset, m, n, seed, method):

| Field | Meaning |
|---|---|
| `method`, `seed`, `dataset`, `m`, `n`, `services` | case identity; `dataset` is `syn`, `wsd` or `synreg`; `services` = m·n |
| `phase`, `group`, `budget` | protocol phase, experiment group, fitness-evaluation budget |
| `switch_ms` | handshake latency per rebinding (ms); absent in the first 735 records of `test_main`, which used the default 40 ms |
| `fit_plan` | penalized fitness of the returned binding under the model the method planned with |
| `fit_markov`, `V_plan_markov` | penalized fitness and constraint violation of that binding under the undiscounted learned Markov model |
| `evals` | fitness evaluations used |
| `t_opt`, `t_runtime` | seconds spent optimizing, and building shortlists and MDP policies |
| `lat`, `lat_p95` | executed end-to-end latency (ms, including handshakes): mean and 95th percentile over slots |
| `rel` | fraction of slots in which every role succeeded |
| `thr`, `cost`, `eng` | executed bottleneck throughput, cost and energy (mJ) |
| `deadline`, `solved` | fraction of slots within the latency threshold; fraction that also succeeded |
| `switches` | rebindings per session |
| `U`, `Vexec` | executed utility and constraint violation |
| `history` | `[evaluations used, best penalized fitness so far]` after each evaluated batch |
| `milp_status`, `milp_status_code`, `milp_time`, `milp_gap` | MILP only; `milp_status_code` (SciPy result code, 0 = optimal) is stored by this version only |

## Reproducibility

**Tables and figures from the shipped records: exact.** With the pinned environment:

- `analyze.py all` reproduces every file in `paper/tables/` and every `results/summary_*.csv` byte
  for byte;
- `figures_pro.py` reproduces the figures (the PDFs carry no timestamp);
- CI regenerates the tables on every push and fails if any of them differs.

**New runs: deterministic.** Instances, learned models and execution use fixed seeds. The optimizer
random stream is seeded with the seed and `zlib.crc32` of the method name, so a run gives the same
records in any process. Two things depend on the machine:

- timing fields;
- MILP runs that stop at the 10 s wall-clock limit: 46 of the 420 MILP runs in `test_main` and 18
  of 60 in `test_regime`.

**Known limitation of the shipped records.** The shipped records of the population-based
optimizers were generated with an earlier derivation of the optimizer random seed, so this version
cannot regenerate them bit for bit. This covers MDP-BWO, BWO, GA, PSO, GWO, ACO, the `+MF` and
`+R0` variants, every ablation variant except A12, and the tuning trials in
`results/tuning.jsonl.gz`.

- **What does reproduce exactly:** the deterministic runs, i.e. MILP runs solved to optimality and
  ablation variant A12 (greedy + MDP). `scripts/check_repro.py` verifies this, and CI runs it.
- **Effect of a re-run with the frozen hyperparameters:** re-running all population-based methods
  of the main comparison with this version changed pooled executed utility by at most about 0.005.
- **Effect of re-tuning first:** the tuning run reproduces every trial setting, but its scores
  differ, and the selected settings of BWO, GA, PSO and GWO change. Re-running those four
  baselines with the new settings changed pooled executed utility by at most about 0.006.
- In both cases no significance or best-method mark in `main_results.tex` changed, but individual
  table entries move in the third decimal.
- **To align the records with this version:** re-run the test phase (for full consistency, the
  tuning first) and regenerate the outputs; `check_repro.py --all` then passes for every method.

```bash
python scripts/check_repro.py --group main --config synthetic:5:20 --seed 1000        # deterministic methods
python scripts/check_repro.py --group main --config synthetic:5:20 --seed 1000 --all  # every method
```

## Citation

Please cite the manuscript (see also `CITATION.cff`); the entry will be updated on publication:

```bibtex
@unpublished{sefati2026choreography,
  title   = {Service Choreography in the Internet of Things Based on Forecast-Guided Binding and Local Rebinding},
  author  = {Sefati, Seyed Salar and Vlahov, Atanas and Craciunescu, Razvan and Arasteh, Bahman and Poulkov, Vladimir},
  year    = {2026},
  note    = {Manuscript under review}
}
```

and, if you use the WS-DREAM data:

```bibtex
@article{zheng2014investigating,
  title   = {Investigating {QoS} of Real-World Web Services},
  author  = {Zheng, Zibin and Zhang, Yilei and Lyu, Michael R.},
  journal = {IEEE Transactions on Services Computing},
  volume  = {7},
  number  = {1},
  pages   = {32--39},
  year    = {2014},
  doi     = {10.1109/TSC.2012.34}
}
```

## License

- **Code:** MIT License (`LICENSE`), © 2026 Seyed Salar Sefati, Seyedeh Tina Sefati, Atanas Vlahov,
  Razvan Craciunescu, Bahman Arasteh and Vladimir Poulkov.
- **Data:** the files in `data/` are third-party data under the WS-DREAM project's MIT License
  (`data/LICENSE`).

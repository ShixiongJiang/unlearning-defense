# Unlearning Defense for Poisoned Offline Safe RL

Code for poisoning offline safe-RL datasets and unlearning the poisoned behavior
from trained safe-RL policies (Safe-RULE). Built on
[OSRL](https://github.com/liuzuxin/OSRL) and [DSRL](https://github.com/liuzuxin/DSRL).

Supported algorithms: BCQL, CPQ, BEARL, BC, CDT, COptiDICE.

## Setup

```bash
conda create -n unlearning python=3.10 && conda activate unlearning
pip install -r requirements.txt

# OSRL is vendored (not tracked); clone it at the tested commit and apply the local fixes.
git clone https://github.com/liuzuxin/OSRL.git
git -C OSRL checkout 03a4586463690566ec8d7830ca4027b46d4ad02f
git -C OSRL apply ../patches/osrl-local-changes.patch

# Run everything from the repository root with both the repo and OSRL importable.
export PYTHONPATH="$PWD:$PWD/OSRL${PYTHONPATH:+:$PYTHONPATH}"
```

`patches/osrl-local-changes.patch` makes the CPQ `log_alpha` and COptiDICE dual
variables persist in checkpoints, makes the CPQ OOD cost regularizer differentiate
the online cost critic, and loads checkpoints onto CPU before moving them.

## Layout

```
configs/                 Per-algorithm training configs
utils/                   Library code
  unlearn_utils.py         Per-algorithm unlearning trainers
  unlearn_config.py        Shared unlearning hyperparameters (UnlearnHyperparameters)
  unlearning_hyperparameters.md  Meaning of every unlearning hyperparameter
  poison_utils.py          Dataset poisoning attacks
  relabel_unlearn_utils.py Relabel-baseline helpers
scripts/                 Entry points (run from the repository root)
  train/                   train_{bc,bcql,bearl,cdt,coptidice,cpq}.py: clean policies
                           train_all_tasks.py: launch training over all DSRL tasks
  poison/                  poison_offline_ds.py: build poisoned datasets
                           train_poison_policy_*.py: train on poisoned datasets
                           train_adv_policy.py: PPO adversary for the traj_inject attack
  unlearn/                 unlearn_policy.py: unlearning runner for all six algorithms
  eval/                    eval_*.py: evaluate saved policies
  plot/                    draw.py: plot unlearning results
tests/                   CPU regression tests
patches/                 Local fixes to the vendored OSRL
```

Outputs (`logs/`, `save_data/`, `wandb/`, `job_dir/`, unlearning logs) and
cluster job files (`*.job`) are written relative to the repository root and are not tracked.

## Workflow

```bash
# 1. Train a clean policy
python scripts/train/train_bcql.py --task OfflineCarCircle-v0

# 2. Poison the dataset
python scripts/poison/poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_cost --poison_frac 0.10 \
    --norm linf --eps 20 --steps 10 --batch_size 64 \
    --save_path save_data/offline_car_circle_poisoned_maxcost.npz \
    --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

# 3. Train a policy on the poisoned data
python scripts/poison/train_poison_policy_bcql.py --task OfflineCarCircle-v0 --attack_type max_cost

# 4. Unlearn the poisoned data
python scripts/unlearn/unlearn_policy.py --algo BCQL --unlearn_path logs/.../BCQL_xxx --eval_episodes 20

# 5. Evaluate a saved policy
python scripts/eval/eval_bcql.py --path logs/.../BCQL_xxx --eval_episodes 20
```

`--unlearn_method` selects `safe` (default), `reward-only`, `fine-tune`, or `relabel`.
See the docstrings at the top of each script for the full set of options.

## Tests

```bash
python -m unittest discover -s tests
```

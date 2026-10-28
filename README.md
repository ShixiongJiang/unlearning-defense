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
export PYTHONPATH="$PWD/OSRL:$PYTHONPATH"
```

`patches/osrl-local-changes.patch` makes the CPQ `log_alpha` and COptiDICE dual
variables persist in checkpoints, makes the CPQ OOD cost regularizer differentiate
the online cost critic, and loads checkpoints onto CPU before moving them.

## Layout

| Path | Purpose |
| --- | --- |
| `train_{bc,bcql,bearl,cdt,coptidice,cpq}.py` | Train clean policies |
| `poison_offline_ds.py` | Build poisoned datasets (`max_cost`, `max_reward`, `simple`, `baffle`, ...) |
| `train_poison_policy_*.py` | Train policies on poisoned datasets |
| `train_adv_policy.py` | PPO adversary used by some attacks |
| `unlearn_policy.py` | Unlearning runner for all six algorithms |
| `eval_*.py`, `examples/eval/` | Evaluate saved policies |
| `examples/configs/` | Per-algorithm training configs |
| `utils/unlearn_utils.py` | Per-algorithm unlearning trainers |
| `utils/unlearn_config.py` | Shared unlearning hyperparameters (`UnlearnHyperparameters`) |
| `utils/unlearning_hyperparameters.md` | Meaning of every unlearning hyperparameter |
| `utils/poison_utils.py`, `utils/relabel_unlearn_utils.py` | Poisoning and relabel-baseline helpers |
| `draw.py` | Plot unlearning curves |
| `tests/` | CPU regression tests |

All scripts are run from the repository root. Outputs (`logs/`, `save_data/`,
`wandb/`, `job_dir/`, unlearning logs) and cluster job files (`*.job`) are not tracked.

## Workflow

```bash
# 1. Train a clean policy
python train_bcql.py --task OfflineCarCircle-v0

# 2. Poison the dataset
python poison_offline_ds.py --task OfflineCarCircle-v0 --attack_type max_cost --poison_frac 0.10 \
    --norm linf --eps 20 --steps 10 --batch_size 64 \
    --save_path save_data/offline_car_circle_poisoned_maxcost.npz \
    --path logs/OfflineCarCircle-v0-cost-10/clean_BCQL/BCQL

# 3. Train a policy on the poisoned data
python train_poison_policy_bcql.py --task OfflineCarCircle-v0 --attack_type max_cost

# 4. Unlearn the poisoned data
python unlearn_policy.py --algo BCQL --unlearn_path logs/.../BCQL_xxx --eval_episodes 20
```

`--unlearn_method` selects `safe` (default), `reward-only`, `fine-tune`, or `relabel`.
See the docstrings at the top of each script for the full set of options.

## Tests

```bash
PYTHONPATH="$PWD/OSRL:$PWD" python -m unittest discover -s tests
```

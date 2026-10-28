# Unlearning hyperparameters

`UnlearnHyperparameters` in `utils/unlearn_config.py` is the single definition
of Safe-RULE defaults. The command-line configuration uses those defaults,
validates them once, and attaches the resulting configuration to every trainer.
A direct per-step argument overrides that trainer configuration; an omitted
argument preserves it.

| Shared setting | Default | Meaning |
| --- | --- | --- |
| `alpha_pg_keep` | 1.0 | Retained-data actor objective weight |
| `cost_forget_scale` | 1.0 | Explicit forget-cost term or the algorithm's documented proxy |
| `keep_bc_weight` | 1.0 | Native behavior-retention term: imitation, MMD support, or weighted likelihood |
| `max_grad_norm` | 1.0 | Phase-one gradient clipping; zero disables clipping |
| `eta_mode` | `adaptive` | Adaptive or fixed forgetting-gradient multiplier |
| `eta_fixed` | 1.0 | Applied multiplier in fixed mode |
| `eta_initial` | 0.1 | Initial adaptive multiplier |
| `eta_min`, `eta_max` | 0.01, 1.0 | Adaptive multiplier bounds |
| `eta_beta` | 0.9 | Adaptive multiplier EMA coefficient |
| `eta_temperature` | 0.25 | Gap-to-weight sigmoid temperature |
| `grad_norm_beta` | 0.95 | Gradient-norm EMA coefficient |
| `grad_norm_floor` | 0.001 | Minimum gradient-normalization denominator |
| `fisher_beta` | 0.99 | Retained-gradient squared EMA coefficient |
| `fisher_damping` | 0.001 | Forget-gradient preconditioner damping |
| `fisher_clip` | 0.05 | Componentwise preconditioned forget-gradient clipping |

The shared gradient update is retained gradient plus eta times the normalized,
projected, and Fisher-scaled forget gradient. All six trainers honor fixed eta,
adaptive eta, and explicit forget-cost scaling. `reward-only` sets the explicit
forget-cost scale to zero while preserving native retained-data training.
It does not remove every native cost constraint.

| Setting for Q-critic methods | Default | Applies to |
| --- | --- | --- |
| `alpha_q_keep`, `alpha_q_forget` | 1.0, 0.1 | BCQL, CPQ, BEARL reward-critic updates |
| `alpha_cq_keep`, `alpha_cq_forget` | 1.0, 0.1 | BCQL, CPQ, BEARL cost-critic updates |
| `keep_cost_weight` | 1.0 | BCQL, CPQ, BEARL actor safety penalty |
| `reward_forget_quantile` | 0.30 | Reward-critic suppression reference |
| `reward_forget_temperature` | 0.5 | Reward-critic suppression softplus |
| `cost_forget_temperature` | 0.25 | Cost-critic inflation and actor safety softplus |
| `forget_cost_margin` | 0.5 | Forgotten dataset action's cost-critic target margin |

BC and CDT do not have these Q critics, and COptiDICE uses state-only dual
potentials. Q-critic coefficients and target temperatures therefore have no
corresponding critic update in those three adapters.

| Algorithm | Forgetting signal / adaptive eta gap | Separate target |
| --- | --- | --- |
| BCQL | Predicted cost of policy actions minus native safety threshold | Native discounted cost threshold |
| CPQ | Predicted cost of policy actions minus native `q_thres` | Native policy safety threshold; `qc_thres` remains the separate OOD/inflation reference |
| BEARL | Predicted cost of policy actions minus native safety threshold | Native discounted threshold; native MMD support constraint retained |
| BC | Target action distance minus achieved mean action distance | `bc_forget_distance_target=0.10`, `bc_forget_temperature=0.05` |
| CDT | Target high-cost-class confidence minus achieved confidence | `cdt_forget_cost_confidence_target=0.7` |
| COptiDICE | Mean forgotten-data occupancy weight minus target | `coptidice_forget_weight_target=1.0` |

Positive gaps request a stronger forgetting update. BC displacement, CDT class
confidence, and COptiDICE occupancy weights are different proxies with different
units. They are not environmental trajectory costs. COptiDICE preserves its
native weighted-likelihood update and uses a nonnegative chi-derived weighting
proxy rather than treating a state-only potential as an action-dependent critic.

The runner restores each algorithm's saved optimizer settings unless explicitly
overridden. BC and BEARL no longer fall back silently to constructor learning
rates. CPQ's auxiliary alpha learning rate uses its native constructor default
when older configs omit that field. CDT retains its native Adam betas, scheduler,
auxiliary prediction weights, entropy learning, and phase-two clipping. Matching
shared unlearning hyperparameters does not force different base algorithms to
use identical architectures, learning rates, batch sizes, or native duals.

CDT and COptiDICE loaders respect terminal and timeout boundaries. COptiDICE
batches apply configured reward/cost scaling and derive initialization
frequency from the actual retained sampler, even when source statistics exist.
Persisted observation/action standard deviations are reused where available;
otherwise they are derived from processed retained data with explicit provenance.
BC defaults absent label scales to
one and restores its configured behavior-cloning mode.

Input recordings must carry valid episode or collector-timeout boundaries.
Segments longer than the saved environment horizon are rejected before training.
In particular, the original unflagged Goal injected file needs its verified
500-step collector boundaries; it must
not be treated as one very long episode. A final partial recording is closed
as a timeout rather than silently dropped.

For CDT, reversed cost labels also reverse the unsafe class: forgetting targets
class zero when `cost_reverse=True`, and class one otherwise. The same unsafe
class is used for the adaptive confidence signal. The runner reports raw CDT
environment cost even when the configured cost scale differs from one.

Effective shared settings and optimizer groups are saved in the unlearning run
configuration/protocol. All six adapters are evaluated with the native trainer
evaluation loop.

Validation: the CPU regression suite in `tests/` covers
real-model forgetting and native continuation updates for all six algorithms,
nondefault settings that change actual gradients, effective optimizer groups,
timeout/scaling checks, and native evaluation with nonunit reward/cost scales.
Twelve BCQL/CPQ control combinations also match the earlier corrected default
implementation exactly over three updates (weights, actor gradients, metrics).

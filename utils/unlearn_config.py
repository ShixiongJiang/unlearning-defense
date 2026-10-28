"""Validated defaults shared by every Safe-RULE trainer.

Native algorithm architecture, optimizers, support constraints and targets stay
in their saved model config. Proxy targets below have explicitly different
units and must not be equated to an environment trajectory-cost budget.
"""
from dataclasses import dataclass, fields, replace
import math


@dataclass(frozen=True)
class UnlearnHyperparameters:
    alpha_q_keep: float = 1.0
    alpha_q_forget: float = 0.1
    alpha_cq_keep: float = 1.0
    alpha_cq_forget: float = 0.1
    alpha_pg_keep: float = 1.0
    cost_forget_scale: float = 1.0
    keep_cost_weight: float = 1.0
    keep_bc_weight: float = 1.0
    max_grad_norm: float = 1.0
    eta_mode: str = "adaptive"
    eta_fixed: float = 1.0
    eta_initial: float = 0.1
    eta_beta: float = 0.9
    eta_min: float = 0.01
    eta_max: float = 1.0
    eta_temperature: float = 0.25
    grad_norm_beta: float = 0.95
    grad_norm_floor: float = 1e-3
    fisher_beta: float = 0.99
    fisher_damping: float = 1e-3
    fisher_clip: float = 0.05
    reward_forget_quantile: float = 0.30
    reward_forget_temperature: float = 0.5
    cost_forget_temperature: float = 0.25
    forget_cost_margin: float = 0.5
    bc_forget_distance_target: float = 0.10
    bc_forget_temperature: float = 0.05
    cdt_forget_cost_confidence_target: float = 0.7
    coptidice_forget_weight_target: float = 1.0

    def __post_init__(self):
        if self.eta_mode not in ("adaptive", "fixed"):
            raise ValueError("eta_mode must be adaptive or fixed")
        for field in fields(self):
            if field.name == "eta_mode":
                continue
            value = getattr(self, field.name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if value < 0:
                raise ValueError(f"{field.name} must be nonnegative")
        for name in ("eta_temperature", "grad_norm_floor", "fisher_damping",
                     "reward_forget_temperature", "cost_forget_temperature", "bc_forget_temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("eta_beta", "grad_norm_beta", "fisher_beta"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("reward_forget_quantile", "cdt_forget_cost_confidence_target"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not self.eta_min <= self.eta_initial <= self.eta_max:
            raise ValueError("Require eta_min <= eta_initial <= eta_max")


DEFAULT_UNLEARN = UnlearnHyperparameters()


def resolve_unlearn_hyperparameters(trainer, **overrides):
    """Direct per-step arguments override the trainer's configured defaults."""
    configured = getattr(trainer, "unlearn_config", DEFAULT_UNLEARN)
    if not isinstance(configured, UnlearnHyperparameters):
        raise TypeError("trainer.unlearn_config must be UnlearnHyperparameters")
    values = {key: value for key, value in overrides.items() if value is not None}
    return replace(configured, **values) if values else configured


def configure_unlearning(trainer, hyperparameters):
    if not isinstance(hyperparameters, UnlearnHyperparameters):
        raise TypeError("Expected UnlearnHyperparameters")
    trainer.unlearn_config = hyperparameters
    return trainer

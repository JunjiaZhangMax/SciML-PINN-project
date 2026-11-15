"""JSON configurations and the predeclared 25-run experiment suite."""

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    name: str = "known_n5000_noise0_seed0"
    group: str = "known"
    mode: str = "known"
    n_observations: int = 5000
    noise: float = 0.0
    seed: int = 0
    observation_seed: int = 0
    noise_seed: int = 10000
    split_seed: int = 2026
    collocation_seed: int = 2027
    test_per_time: int = 500
    collocation: str = "observations"
    n_collocation: int = 5000
    # None uses all reference times; e.g. 1.0 starts with a shorter time window.
    time_end: float | None = None
    hidden_layers: int = 8
    width: int = 20
    dtype: str = "float64"
    lambda1_init: float = 0.0
    lambda2_init: float = 0.0
    # Experiment settings; these are not learned PDE coefficients.
    w_observation: float = 1.0
    w_initial: float = 1.0
    w_boundary: float = 1.0
    w_residual: float = 1.0
    w_gauge: float = 1.0
    adam_steps: int = 50000
    adam_lr: float = 0.001
    # Legacy settings retained so existing JSON configs still load; unused by train().
    lbfgs_steps: int = 0
    lbfgs_history: int = 50
    physics_batch_size: int = 1000
    eval_batch_size: int = 4096
    log_every: int = 100
    checkpoint_every: int = 1000
    device: str = "auto"
    purpose: str = "production"

    def validate(self):
        if self.mode not in {"known", "inverse"}:
            raise ValueError("mode must be known or inverse")
        if self.group not in {"known", "reference", "sensitivity"}:
            raise ValueError("Unknown experiment group")
        if self.collocation not in {"observations", "fixed"}:
            raise ValueError("collocation must be observations or fixed")
        if self.dtype != "float64":
            raise ValueError("This protocol uses float64")
        if not math.isfinite(self.noise) or not 0 <= self.noise <= 1:
            raise ValueError("noise must be a finite fraction in [0, 1]")
        if self.time_end is not None and not math.isfinite(self.time_end):
            raise ValueError("time_end must be finite or None")
        for key in ("n_observations", "n_collocation", "test_per_time", "hidden_layers",
                    "width", "physics_batch_size", "eval_batch_size", "log_every",
                    "checkpoint_every", "lbfgs_history"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("seed", "observation_seed", "noise_seed", "split_seed", "collocation_seed",
                    "adam_steps", "lbfgs_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if self.adam_steps == 0:
            raise ValueError("At least one Adam step is required")
        if not math.isfinite(self.adam_lr) or self.adam_lr <= 0:
            raise ValueError("adam_lr must be positive and finite")
        if not all(math.isfinite(v) for v in (self.lambda1_init, self.lambda2_init)):
            raise ValueError("Initial coefficients must be finite")
        for key in ("w_observation", "w_initial", "w_boundary", "w_residual", "w_gauge"):
            weight = getattr(self, key)
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.purpose not in {"production", "smoke", "pilot"}:
            raise ValueError("purpose must be production, smoke, or pilot")
        if not self.name or Path(self.name).name != self.name or any(c in self.name for c in '\\/:*?"<>|'):
            raise ValueError("name must be a safe directory name")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        return self

    def to_dict(self):
        return asdict(self)

    def forward_loss_weights(self) -> dict[str, float]:
        """Keyword arguments for forward_objective; observation loss is not used."""
        return {key: getattr(self, key) for key in
                ("w_initial", "w_boundary", "w_residual", "w_gauge")}

    def inverse_loss_weights(self) -> dict[str, float]:
        """Keyword arguments for inverse_objective, including optional IC/BC terms."""
        return {key: getattr(self, key) for key in
                ("w_observation", "w_initial", "w_boundary", "w_residual")}

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**value).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def suite(base=None):
    """Known check first, then reference baselines, then six unique sparse conditions."""
    base = base or Config()
    result = [replace(base, name="known_n5000_noise0_seed0", group="known", mode="known",
                      n_observations=5000, noise=0, seed=0, observation_seed=0,
                      noise_seed=10000, collocation="observations")]
    for noise in (0, 0.01):
        for seed in (0, 1, 2):
            result.append(replace(base, name=f"reference_n5000_noise{noise:g}_seed{seed}",
                                  group="reference", mode="inverse", n_observations=5000,
                                  noise=noise, seed=seed, observation_seed=seed,
                                  noise_seed=10000 + seed, collocation="observations"))
    for count, noise in ((100, 0), (100, 0.01), (100, 0.05), (100, 0.10), (50, 0.05), (200, 0.05)):
        for seed in (0, 1, 2):
            result.append(replace(base, name=f"sensitivity_n{count}_noise{noise:g}_seed{seed}",
                                  group="sensitivity", mode="inverse", n_observations=count,
                                  noise=noise, seed=seed, observation_seed=seed,
                                  noise_seed=10000 + seed, collocation="fixed"))
    return [c.validate() for c in result]

# pipeline/registration.py
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Dataclasses — représentent une stage ANTs
# ─────────────────────────────────────────

@dataclass
class MetricConfig:
    name: str                        # MI, Mattes, CC, MeanSquares
    weight: float = 1.0
    bins: int = 32
    sampling: str = "Regular"        # Regular, Random, None
    sampling_rate: float = 0.25

    def to_ants_string(self, fixed: str, moving: str) -> str:
        return (
            f"{self.name}[{fixed},{moving},"
            f"{self.weight},{self.bins},"
            f"{self.sampling},{self.sampling_rate}]"
        )


@dataclass
class StageConfig:
    metric: MetricConfig
    transform: str                   # Rigid, Affine, SyN, BSplineSyN...
    transform_params: list[float]
    convergence: list[int]
    shrink_factors: list[int]
    smoothing_sigmas: list[float]
    use_histogram_matching: bool = False

    def __post_init__(self):
        n = len(self.convergence)
        if len(self.shrink_factors) != n or len(self.smoothing_sigmas) != n:
            raise ValueError(
                f"Stage '{self.transform}': convergence, shrink_factors et "
                f"smoothing_sigmas doivent avoir la même longueur ({n})"
            )

    @property
    def transform_string(self) -> str:
        params = ",".join(str(p) for p in self.transform_params)
        return f"{self.transform}[{params}]"

    @property
    def convergence_string(self) -> str:
        return "x".join(str(v) for v in self.convergence)

    @property
    def shrink_string(self) -> str:
        return "x".join(str(v) for v in self.shrink_factors)

    @property
    def smooth_string(self) -> str:
        return "x".join(str(v) for v in self.smoothing_sigmas)


@dataclass
class RegistrationConfig:
    stages: list[StageConfig]
    init_transform: Optional[str] = None
    verbose: bool = True
    dimensionality: int = 3


# ─────────────────────────────────────────
# Parsers YAML → dataclasses
# ─────────────────────────────────────────

def _parse_metric(cfg: dict) -> MetricConfig:
    p = cfg.get("metric_params", {})
    return MetricConfig(
        name=cfg["metric"],
        weight=p.get("weight", 1.0),
        bins=p.get("bins", 32),
        sampling=p.get("sampling", "Regular"),
        sampling_rate=p.get("sampling_rate", 0.25),
    )


def _parse_stage(cfg: dict) -> StageConfig:
    return StageConfig(
        metric=_parse_metric(cfg),
        transform=cfg["transform"],
        transform_params=cfg.get("transform_params", [0.1]),
        convergence=cfg["convergence"],
        shrink_factors=cfg["shrink_factors"],
        smoothing_sigmas=cfg["smoothing_sigmas"],
        use_histogram_matching=cfg.get("use_histogram_matching", False),
    )


def parse_registration_config(reg_cfg: dict) -> RegistrationConfig:
    """
    Reçoit le bloc YAML d'une registration (ref_to_template ou ref_to_qmri).
    Retourne un RegistrationConfig validé.
    """
    stages = [_parse_stage(s) for s in reg_cfg["stages"]]
    return RegistrationConfig(
        stages=stages,
        init_transform=reg_cfg.get("init_transform"),
        verbose=reg_cfg.get("verbose", True),
        dimensionality=reg_cfg.get("dimensionality", 3),
    )


# ─────────────────────────────────────────
# Builder de commande ANTs
# ─────────────────────────────────────────

def _build_command(
    fixed: str,
    moving: str,
    out_prefix: str,
    config: RegistrationConfig,
) -> list[str]:
    warped = f"{out_prefix}_warped.nii.gz"
    inv_warped = f"{out_prefix}_inv_warped.nii.gz"

    cmd = [
        "antsRegistration",
        "-d", str(config.dimensionality),
        "-o", f"[{out_prefix},{warped},{inv_warped}]",
    ]

    if config.init_transform:
        cmd += ["-r", config.init_transform]

    for stage in config.stages:
        cmd += ["-m", stage.metric.to_ants_string(fixed, moving)]
        cmd += ["-t", stage.transform_string]
        cmd += ["-c", f"[{stage.convergence_string},1e-6,10]"]
        cmd += ["-s", stage.smooth_string]
        cmd += ["-f", stage.shrink_string]
        if stage.use_histogram_matching:
            cmd += ["--use-histogram-matching", "1"]

    if config.verbose:
        cmd += ["--verbose", "1"]

    return cmd


# ─────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────

def run_registration(
    fixed: Path,
    moving: Path,
    out_prefix: Path,
    config: RegistrationConfig,
    log_file: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Lance antsRegistration selon la config fournie.
    Retourne les paths des outputs (warped, inv_warped, transforms).
    """
    fixed = Path(fixed)
    moving = Path(moving)
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    log_file = log_file or out_prefix.with_suffix(".log")

    cmd = _build_command(str(fixed), str(moving), str(out_prefix), config)

    logger.info("Lancement ANTs : %s → %s", moving.name, fixed.name)
    logger.debug("Commande : %s", " ".join(cmd))

    with open(log_file, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"antsRegistration a échoué (code {result.returncode}). "
            f"Voir le log : {log_file}"
        )

    return {
        "warped": out_prefix.parent / f"{out_prefix.name}_warped.nii.gz",
        "inv_warped": out_prefix.parent / f"{out_prefix.name}_inv_warped.nii.gz",
        "prefix": out_prefix,
        "log": log_file,
    }
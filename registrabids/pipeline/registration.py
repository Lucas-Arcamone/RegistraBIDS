# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lucas ARCAMONE

import subprocess
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Dataclasses — represent an ANTs stage
# ─────────────────────────────────────────


@dataclass
class MetricConfig:
    name: str  # MI, Mattes, otherwise CC, MeanSquares have radius
    weight: float = 1.0
    bins: int = 32
    sampling: str = "Regular"  # Regular, Random, None
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
    transform: str  # Rigid, Affine, SyN, BSplineSyN...
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
# ANTs Command Builder
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
        "-d",
        str(config.dimensionality),
        "--float",
        "0",
        "-o",
        f"[{out_prefix},{warped},{inv_warped}]",
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


def _ants_env() -> dict:
    """
    Environment variables to force gzip compression in ANTs output.
    ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS: stability
    ANTS_RANDOM_SEED: reproducibility
    """
    env = os.environ.copy()
    env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
    env["ANTS_RANDOM_SEED"] = "42"
    return env


def _verify_nifti(path: Path) -> None:
    """Verifies that a .nii.gz file is a valid gzip file."""
    import gzip

    if not path.exists():
        raise RuntimeError(f"Missing ANTs output file : {path}")
    try:
        with gzip.open(path, "rb") as f:
            f.read(4)  # lit juste le header gzip
    except gzip.BadGzipFile:
        raise RuntimeError(
            f"The ANTs output file is not a valid gzip file: {path}\n"
            f"Check the associated ANTs log."
        )


# ─────────────────────────────────────────
# Fonction principale
# ─────────────────────────────────────────


def run_registration(
    fixed: Path,
    moving: Path,
    out_prefix: Path,
    config: RegistrationConfig,
    log_file: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Run antsRegistration based on the provided configuration.
    Returns the paths of the outputs (warped, inv_warped, transforms).
    """
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    warped = out_prefix.parent / f"{out_prefix.name}_warped.nii.gz"

    # Le recalage est considéré terminé si le warped existe et est non vide
    if not force and warped.exists() and warped.stat().st_size > 0:
        logger.info("Skip recalage — output déjà existant : %s", warped.name)
        return {
            "warped": warped,
            "inv_warped": out_prefix.parent / f"{out_prefix.name}_inv_warped.nii.gz",
            "prefix": out_prefix,
            "log": log_file or out_prefix.with_suffix(".log"),
        }

    fixed = Path(fixed)
    moving = Path(moving)

    log_file = log_file or out_prefix.with_suffix(".log")

    cmd = _build_command(str(fixed), str(moving), str(out_prefix), config)

    logger.info("Run ANTs : %s → %s", moving.name, fixed.name)
    logger.debug("Command : %s", " ".join(cmd))

    with open(log_file, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=_ants_env(),
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"antsRegistration failed (code {result.returncode}). "
            f"See log file : {log_file}"
        )

    # Vérifie que les fichiers de sortie sont bien là et lisibles
    warped = out_prefix.parent / f"{out_prefix.name}_warped.nii.gz"
    _verify_nifti(warped)

    return {
        "warped": out_prefix.parent / f"{out_prefix.name}_warped.nii.gz",
        "inv_warped": out_prefix.parent / f"{out_prefix.name}_inv_warped.nii.gz",
        "prefix": out_prefix,
        "log": log_file,
    }

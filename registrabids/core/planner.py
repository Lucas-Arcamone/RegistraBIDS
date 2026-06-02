# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lucas ARCAMONE

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────
# Dataclasses — work units
# ─────────────────────────────────────────


@dataclass
class RegistrationJob:
    """Un recalage à calculer : fixed → moving → transform."""

    fixed: Path
    moving: Path
    out_prefix: Path
    job_type: str  # "ref_to_template" | "source_to_ref"
    source_key: str  # nom court pour logs ex: "acq-refMT_run-02_res-100iso"


@dataclass
class ApplyTransformJob:
    """Application de la chaîne de transforms sur une qmap."""

    qmap: Path
    source_key: str  # doit matcher un RegistrationJob.source_key
    out_path: Path
    space_ref: Path
    interpolator: str = "Linear"


# ─────────────────────────────────────────
# Dataclasses — jobs de preprocessing
# ─────────────────────────────────────────


@dataclass
class VolumeExtractionJob:
    input_path: Path
    output_path: Path  # temp ou persistant selon save_intermediates
    strategy: str  # mean | first_volume | geometric_mean_shell | weighted_mean_echo
    params: dict = field(default_factory=dict)
    save: bool = False


@dataclass
class DenoisingJob:
    input_path: Path
    output_path: Path
    method: str  # NLMF | MPPCA
    noise_model: str  # rician | gaussian
    patch_radius: int = 1
    search_radius: int = 3
    save: bool = False


@dataclass
class N4Job:
    input_path: Path
    output_path: Path
    shrink_factor: int = 4
    n_iterations: list[int] = field(default_factory=lambda: [50, 50, 30, 20])
    convergence_threshold: float = 0.001
    save: bool = False


@dataclass
class PreprocessingPlan:
    """Plan de preprocessing pour un fichier source ou référence."""

    source_key: str  # "ref" | source_key de la source
    original_path: Path
    jobs: list  # [VolumeExtractionJob?, DenoisingJob?, N4Job?]
    final_path: Path  # path du fichier préprocessé final


# ─────────────────────────────────────────
# Dataclasses — Final Session
# ─────────────────────────────────────────


@dataclass
class SessionPlan:
    """Plan complet pour un (sub, ses)."""

    subject: str
    session: str
    ref: Path
    template: Path
    registration_jobs: list[RegistrationJob] = field(default_factory=list)
    apply_jobs: list[ApplyTransformJob] = field(default_factory=list)
    preprocessing_plans: dict[str, PreprocessingPlan] = field(default_factory=dict)

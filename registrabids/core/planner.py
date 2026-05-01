from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────
# Dataclasses — unités de travail
# ─────────────────────────────────────────

@dataclass
class RegistrationJob:
    """Un recalage à calculer : fixed → moving → transform."""
    fixed: Path
    moving: Path
    out_prefix: Path
    job_type: str          # "ref_to_template" | "source_to_ref"
    source_key: str        # nom court pour logs ex: "acq-refMT_run-02_res-100iso"


@dataclass
class ApplyTransformJob:
    """Application de la chaîne de transforms sur une qmap."""
    qmap: Path
    source_key: str        # doit matcher un RegistrationJob.source_key
    out_path: Path


@dataclass
class SessionPlan:
    """Plan complet pour un (sub, ses)."""
    subject: str
    session: str
    ref: Path
    template: Path
    registration_jobs: list[RegistrationJob] = field(default_factory=list)
    apply_jobs: list[ApplyTransformJob] = field(default_factory=list)
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lucas ARCAMONE
from __future__ import annotations
import logging
import os
from pathlib import Path

from joblib import Parallel, delayed

from registrabids.core.planner import SessionPlan
from registrabids.pipeline.preprocessing import run_preprocessing_plan
from registrabids.pipeline.registration import parse_registration_config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# Resource allocation
# ─────────────────────────────────────────


def resolve_parallelism(config: dict) -> tuple[int, int]:
    """
    Determined (n_sessions, n_workers) based on the configuration or available resources.

    Priority:
    1. disabled: true  → sequential (1, 1)
    2. Explicit values in the YAML
    3. Automatic based on allocated CPUs (Slurm-aware via sched_getaffinity)
    """
    parallelism_cfg = config.get("parallelism", {})

    if parallelism_cfg.get("disabled", False):
        logger.info("Parallelization explicitly disabled → sequential mode.")
        return 1, 1

    n_sessions = parallelism_cfg.get("n_sessions")
    n_workers = parallelism_cfg.get("n_workers")

    if n_sessions is not None and n_workers is not None:
        logger.info(
            "Explicite parallelism : %d session(s) × %d worker(s).",
            n_sessions,
            n_workers,
        )
        return int(n_sessions), int(n_workers)

    available_cpus = _available_cpus()
    n_sessions = 1
    n_workers = max(1, available_cpus)
    logger.info(
        "Automatic parallelism : %d CPU(s) detected → %d session(s) × %d worker(s).",
        available_cpus,
        n_sessions,
        n_workers,
    )
    return n_sessions, n_workers


def _available_cpus() -> int:
    """
    Returns the number of available CPUs.
    os.sched_getaffinity(0) respects Slurm allocations.
    Falls back to os.cpu_count() if unavailable (macOS, Windows).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# ─────────────────────────────────────────
# Helpers — reconstruction après barrières
# ─────────────────────────────────────────


def _collect_preprocessed(plan: SessionPlan) -> dict[str, Path]:
    """
    Rebuilds {source_key: final_path} from the executed PreprocessingPlans.
    Called after the preprocessing barrier to feed the re-alignments.
    """
    result = {
        key: preproc_plan.final_path
        for key, preproc_plan in plan.preprocessing_plans.items()
    }
    result.setdefault("ref", plan.ref)
    return result


def _collect_transform_prefixes(plan: SessionPlan) -> dict[str, Path]:
    """
    Reconstructs {source_key: out_prefix} from the executed RegistrationJobs.
    Called after the registration barrier to feed the apply transforms.
    Checks that the transform files exist.
    """
    prefixes = {}
    errors = []

    for job in plan.registration_jobs:
        prefix = job.out_prefix
        affine = prefix.parent / f"{prefix.name}0GenericAffine.mat"

        if not affine.exists():
            errors.append(f"Missing transform for '{job.source_key}' : {affine}")
            continue

        prefixes[job.source_key] = prefix
        logger.debug(
            "Collected prefix [%s] : %s",
            job.source_key,
            prefix.name,
        )

    if errors:
        raise RuntimeError(
            f"{len(errors)} missing transform(s) after realignment :\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    return prefixes


# ─────────────────────────────────────────
# Exécution parallèle d'une session
# ─────────────────────────────────────────


def run_session_parallel(
    plan: SessionPlan,
    reg_config_template: dict,
    reg_config_qmri: dict,
    force: bool = False,
    n_workers: int = 1,
) -> None:
    """
    Executes the entire session plan with intra-session parallelism.

    Three sequential steps separated by implicit barriers
    (joblib waits for all jobs to finish before moving on to the next step):

      1. Preprocessing  — all preprocessing runs in parallel
      2. Registration   — all registration runs in parallel
      3. Apply          — all apply transforms run in parallel
    """
    config_template = parse_registration_config(reg_config_template)
    config_qmri = parse_registration_config(reg_config_qmri)

    logger.info(
        "Session sub-%s ses-%s — %d preproc | %d recalages | %d apply",
        plan.subject,
        plan.session,
        len(plan.preprocessing_plans),
        len(plan.registration_jobs),
        len(plan.apply_jobs),
    )

    # ── Step 1 : preprocessing ─────────────────────────────────────────
    logger.info("[sub-%s ses-%s] Step 1/3 : preprocessing", plan.subject, plan.session)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(run_preprocessing_plan)(preproc_plan, force)
        for preproc_plan in plan.preprocessing_plans.values()
    )

    # ── Barrier 1 → reconstruction of preprocessed paths ───────────────
    preprocessed = _collect_preprocessed(plan)

    # ── Step 2 : registrations ─────────────────────────────────────────
    logger.info("[sub-%s ses-%s] Step 2/3 : registration", plan.subject, plan.session)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_run_registration_job)(
            job=job,
            preprocessed=preprocessed,
            config_template=config_template,
            config_qmri=config_qmri,
            force=force,
        )
        for job in plan.registration_jobs
    )

    # ── Barrier 2 → reconstruction of transform prefixes ───────────
    transform_prefixes = _collect_transform_prefixes(plan)

    # ── Step 3 : apply transforms ───────────────────────────────────────
    logger.info(
        "[sub-%s ses-%s] Step 3/3 : apply transforms", plan.subject, plan.session
    )

    apply_args = _build_apply_args(plan, transform_prefixes)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_apply_one)(job, transforms) for job, transforms in apply_args
    )

    logger.info(
        "Session sub-%s ses-%s Done — %d qmap(s) in the template space.",
        plan.subject,
        plan.session,
        len(plan.apply_jobs),
    )


# ─────────────────────────────────────────
# Private helpers — unit tests
# ─────────────────────────────────────────


def _run_registration_job(
    job,
    preprocessed: dict[str, Path],
    config_template,
    config_qmri,
    force: bool,
) -> None:
    """Runs a single RegistrationJob using the preprocessed paths."""
    from registrabids.pipeline.registration import run_registration

    if job.job_type == "ref_to_template":
        fixed = job.fixed
        moving = preprocessed.get("ref", job.moving)
        cfg = config_template
    else:
        fixed = preprocessed.get("ref", job.fixed)
        moving = preprocessed.get(job.source_key, job.moving)
        cfg = config_qmri

    run_registration(
        fixed=fixed,
        moving=moving,
        out_prefix=job.out_prefix,
        config=cfg,
        force=force,
    )


def _build_apply_args(
    plan: SessionPlan,
    transform_prefixes: dict[str, Path],
) -> list[tuple]:
    """
    Builds the list of (ApplyTransformJob, transforms) for step 3.
    Filters out jobs whose source_key does not have an associated transform.
    """
    args = []
    prefix_template = transform_prefixes["ref"]

    for app_job in plan.apply_jobs:
        prefix_source = transform_prefixes.get(app_job.source_key)
        if prefix_source is None:
            logger.error(
                "No transform for source_key='%s' — qmap ignored : %s",
                app_job.source_key,
                app_job.qmap.name,
            )
            continue

        transforms = [
            f"{prefix_template}1Warp.nii.gz",
            f"{prefix_template}0GenericAffine.mat",
            f"{prefix_source}0GenericAffine.mat",
        ]
        args.append((app_job, transforms))
    return args


def _apply_one(job, transforms):
    """A joblib-compatible wrapper for _apply_transforms."""
    from registrabids.pipeline.runner import _apply_transforms

    _apply_transforms(job, transforms)

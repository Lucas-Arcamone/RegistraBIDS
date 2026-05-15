# core/parallelism.py
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
# Résolution des ressources
# ─────────────────────────────────────────


def resolve_parallelism(config: dict) -> tuple[int, int]:
    """
    Résout (n_sessions, n_workers) depuis la config ou les ressources disponibles.

    Priorité :
    1. disabled: true  → séquentiel (1, 1)
    2. Valeurs explicites dans le YAML
    3. Automatique depuis les CPUs alloués (Slurm-aware via sched_getaffinity)
    """
    parallelism_cfg = config.get("parallelism", {})

    if parallelism_cfg.get("disabled", False):
        logger.info("Parallélisation désactivée explicitement → mode séquentiel.")
        return 1, 1

    n_sessions = parallelism_cfg.get("n_sessions")
    n_workers = parallelism_cfg.get("n_workers")

    if n_sessions is not None and n_workers is not None:
        logger.info(
            "Parallélisme explicite : %d session(s) × %d worker(s).",
            n_sessions,
            n_workers,
        )
        return int(n_sessions), int(n_workers)

    available_cpus = _available_cpus()
    n_sessions = 1
    n_workers = max(1, available_cpus)
    logger.info(
        "Parallélisme automatique : %d CPU(s) détecté(s) → "
        "%d session(s) × %d worker(s).",
        available_cpus,
        n_sessions,
        n_workers,
    )
    return n_sessions, n_workers


def _available_cpus() -> int:
    """
    Retourne le nombre de CPUs disponibles.
    os.sched_getaffinity(0) respecte les allocations Slurm.
    Fallback sur os.cpu_count() si non disponible (macOS, Windows).
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
    Reconstruit {source_key: final_path} depuis les PreprocessingPlan exécutés.
    Appelé après la barrière preprocessing pour alimenter les recalages.
    """
    result = {
        key: preproc_plan.final_path
        for key, preproc_plan in plan.preprocessing_plans.items()
    }
    result.setdefault("ref", plan.ref)
    return result


def _collect_transform_prefixes(plan: SessionPlan) -> dict[str, Path]:
    """
    Reconstruit {source_key: out_prefix} depuis les RegistrationJob exécutés.
    Appelé après la barrière registration pour alimenter les apply transforms.
    Vérifie que les fichiers de transform existent.
    """
    prefixes = {}
    errors = []

    for job in plan.registration_jobs:
        prefix = job.out_prefix
        affine = prefix.parent / f"{prefix.name}0GenericAffine.mat"

        if not affine.exists():
            errors.append(f"Transform manquant pour '{job.source_key}' : {affine}")
            continue

        prefixes[job.source_key] = prefix
        logger.debug(
            "Transform prefix collecté [%s] : %s",
            job.source_key,
            prefix.name,
        )

    if errors:
        raise RuntimeError(
            f"{len(errors)} transform(s) manquant(s) après recalage :\n"
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
    Exécute le plan complet d'une session avec parallélisme intra-session.

    Trois étapes séquentielles séparées par des barrières implicites
    (joblib attend que tous les jobs soient finis avant de passer à l'étape suivante) :

      1. Preprocessing  — tous les preproc en parallèle
      2. Registration   — tous les recalages en parallèle
      3. Apply          — tous les apply transforms en parallèle
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

    # ── Étape 1 : preprocessing ──────────────────────────────────────────
    logger.info("[sub-%s ses-%s] Étape 1/3 : preprocessing", plan.subject, plan.session)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(run_preprocessing_plan)(preproc_plan, force)
        for preproc_plan in plan.preprocessing_plans.values()
    )

    # ── Barrière 1 → reconstruction des paths préprocessés ───────────────
    preprocessed = _collect_preprocessed(plan)

    # ── Étape 2 : recalages ──────────────────────────────────────────────
    logger.info("[sub-%s ses-%s] Étape 2/3 : recalages", plan.subject, plan.session)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_run_registration_job)(
            job=job,
            preprocessed=preprocessed,
            config_template=config_template,
            config_qmri=config_qmri,
            n_workers=n_workers,
            force=force,
        )
        for job in plan.registration_jobs
    )

    # ── Barrière 2 → reconstruction des prefixes de transforms ───────────
    transform_prefixes = _collect_transform_prefixes(plan)

    # ── Étape 3 : apply transforms ───────────────────────────────────────
    logger.info(
        "[sub-%s ses-%s] Étape 3/3 : apply transforms", plan.subject, plan.session
    )

    apply_args = _build_apply_args(plan, transform_prefixes)

    Parallel(n_jobs=n_workers, backend="loky")(
        delayed(_apply_one)(job, transforms) for job, transforms in apply_args
    )

    logger.info(
        "Session sub-%s ses-%s terminée — %d qmap(s) dans l'espace template.",
        plan.subject,
        plan.session,
        len(plan.apply_jobs),
    )


# ─────────────────────────────────────────
# Helpers privés — jobs unitaires
# ─────────────────────────────────────────


def _run_registration_job(
    job,
    preprocessed: dict[str, Path],
    config_template,
    config_qmri,
    n_workers: int,
    force: bool,
) -> None:
    """Exécute un RegistrationJob unique avec les paths préprocessés."""
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
    Construit la liste des (ApplyTransformJob, transforms) pour l'étape 3.
    Filtre les jobs dont le source_key n'a pas de transform associé.
    """
    args = []
    prefix_template = transform_prefixes["ref"]

    for app_job in plan.apply_jobs:
        prefix_source = transform_prefixes.get(app_job.source_key)
        if prefix_source is None:
            logger.error(
                "Pas de transform pour source_key='%s' — qmap ignorée : %s",
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
    """Wrapper pour _apply_transforms compatible joblib."""
    from registrabids.pipeline.runner import _apply_transforms

    _apply_transforms(job, transforms)

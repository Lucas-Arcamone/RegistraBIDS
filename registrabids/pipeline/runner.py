import logging
import nibabel as nib
import numpy as np
import tempfile
from pathlib import Path
from joblib import Parallel, delayed
import shutil

from registrabids.core.bids_index import BIDSIndex
from registrabids.core.resolver import ReferenceResolver, RegistrationPlanner
from registrabids.core.planner import SessionPlan, ApplyTransformJob
from registrabids.pipeline.registration import (
    run_registration,
    parse_registration_config,
)
from registrabids.core.template import TemplateLoader
from registrabids.pipeline.preprocessing import run_preprocessing_plan
from registrabids.core.parallelism import resolve_parallelism, run_session_parallel

logger = logging.getLogger(__name__)


def _make_reference_grid(
    template_path: Path,
    qmap_path: Path,
    tmp_dir: Path,
) -> Path:
    """
    Creates a reference image for antsApplyTransforms using:
    - the template's space and orientation
    - the qmap's resolution

    This prevents qmaps from being oversampled at the template's resolution.
    """
    tpl = nib.load(template_path)
    qmap = nib.load(qmap_path)

    tpl_affine = tpl.affine
    tpl_shape = np.array(tpl.shape[:3])
    tpl_voxsize = np.sqrt((tpl_affine[:3, :3] ** 2).sum(axis=0))

    qmap_affine = qmap.affine
    qmap_voxsize = np.sqrt((qmap_affine[:3, :3] ** 2).sum(axis=0))

    # new shape : adjust the template FOV with the qmap resolution
    new_shape = np.round(tpl_shape * (tpl_voxsize / qmap_voxsize)).astype(int)

    # new affine matrix : same origin and orientation as the template,
    # but qmap voxel size
    scaling = np.diag(qmap_voxsize / tpl_voxsize)
    new_affine = tpl_affine.copy()
    new_affine[:3, :3] = tpl_affine[:3, :3] @ scaling

    logger.debug(
        "Hybrid grid : shape %s → %s | voxsize %s → %s",
        tuple(tpl_shape),
        tuple(new_shape),
        np.round(tpl_voxsize, 3).tolist(),
        np.round(qmap_voxsize, 3).tolist(),
    )

    ref_img = nib.Nifti1Image(
        np.zeros(new_shape, dtype=np.float32),
        affine=new_affine,
    )

    ref_path = tmp_dir / f"ref_grid_{qmap_path.stem}.nii.gz"
    nib.save(ref_img, ref_path)
    return ref_path


def _apply_transforms(job: ApplyTransformJob, transforms: list[str]) -> None:
    """
    Call antsApplyTransforms to map the qmap into the template space.
    transforms: an ordered list of transform files to chain together,
                 from newest to oldest (ANTs convention).
    """
    if job.out_path.exists() and job.out_path.stat().st_size > 0:
        logger.info("Skip apply — output déjà existant : %s", job.out_path.name)
        return

    import subprocess

    job.out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix="registrabids_ref_"))

    try:
        ref_grid = _make_reference_grid(
            template_path=job.space_ref,
            qmap_path=job.qmap,
            tmp_dir=tmp_dir,
        )
        cmd = [
            "antsApplyTransforms",
            "-d",
            "3",
            "-e",
            "3",
            "-i",
            str(job.qmap),
            "-r",
            str(ref_grid),
            "-o",
            str(job.out_path),
            "--interpolation",
            "Linear",
        ]
        for t in transforms:
            cmd += ["-t", t]

        logger.info("Applying transforms → %s", job.out_path.name)
        logger.debug("Command : %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"antsApplyTransforms a échoué pour {job.qmap.name} :\n{result.stderr}"
            )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def run_session(
    plan: SessionPlan,
    reg_config_template: dict,  # bloc YAML registration.ref_to_template
    reg_config_qmri: dict,  # bloc YAML registration.ref_to_qmri
    force: bool = False,
) -> None:
    """
    Execute the complete plan for a session :
      1. Registration (ref→template, sources→ref)
      2. Applying successive transforms to each qmap
    """
    config_template = parse_registration_config(reg_config_template)
    config_qmri = parse_registration_config(reg_config_qmri)

    # Preprocessing
    preprocessed: dict[str, Path] = {}

    for source_key, preproc_plan in plan.preprocessing_plans.items():
        preprocessed[source_key] = run_preprocessing_plan(preproc_plan)

    # Fallback si pas de preprocessing configuré
    preprocessed.setdefault("ref", plan.ref)

    # Stocke les prefixes de output par source_key pour récupérer les transforms
    transform_prefixes: dict[str, Path] = {}

    for job in plan.registration_jobs:
        if job.job_type == "ref_to_template":
            fixed = job.fixed  # template — inchangé, jamais préprocessé
            moving = preprocessed.get("ref", job.moving)  # ref préprocessée
        else:
            # source_to_ref
            fixed = preprocessed.get("ref", job.fixed)  # ref préprocessée
            moving = preprocessed.get(job.source_key, job.moving)  # source préprocessée

        cfg = config_template if job.job_type == "ref_to_template" else config_qmri
        result = run_registration(
            fixed=fixed,
            moving=moving,
            out_prefix=job.out_prefix,
            config=cfg,
            force=force,
        )
        transform_prefixes[job.source_key] = result["prefix"]
        logger.info("✓ %s done", job.source_key)

    # ── Application des transforms ───────────────────────────────────────
    prefix_template = transform_prefixes["ref"]

    for app_job in plan.apply_jobs:
        prefix_source = transform_prefixes.get(app_job.source_key)
        if prefix_source is None:
            logger.error(
                "No transform found for source_key='%s', qmap ignored : %s",
                app_job.source_key,
                app_job.qmap.name,
            )
            continue

        # ANTs order: from newest to oldest
        # T(ref→template) ∘ T(source→ref)
        transforms = [
            f"{prefix_template}1Warp.nii.gz",  # warp SyN
            f"{prefix_template}0GenericAffine.mat",  # affine ref→template
            f"{prefix_source}0GenericAffine.mat",  # affine source→ref
        ]
        _apply_transforms(app_job, transforms)

    logger.info(
        "Session sub-%s ses-%s done — %d qmaps in the template space.",
        plan.subject,
        plan.session,
        len(plan.apply_jobs),
    )


def run_pipeline(
    bids_root: str, config: dict, output_dir: Path | None = None, force: bool = False
) -> None:
    """
    Main entry point.
    config: a dictionary derived from the full YAML file.
    """
    output_root = (
        Path(output_dir)
        if output_dir
        else Path(bids_root) / "derivatives" / "registrabids"
    )

    # (Joblib) Parallelism
    n_sessions, n_workers = resolve_parallelism(config)

    index = BIDSIndex(bids_root)
    resolver = ReferenceResolver(index.layout)

    atlas = TemplateLoader.from_config(config["template"])
    template = atlas.template

    # output_root = Path(bids_root) / "derivatives" / "registrabids"
    planner = RegistrationPlanner(index.layout, template, output_root)

    reference_map = resolver.extract_reference_map(config)
    grouped = index.get_qmri_maps_grouped(config)
    source_map = index.map_to_sources(config)

    session_plans = []
    for (sub, ses), qmri_files in grouped.items():
        ref_path = reference_map.get((sub, ses))
        if not ref_path:
            logger.warning("No reference for sub-%s ses-%s, session ignored.", sub, ses)
            continue

        preproc_config = config.get("preprocessing")

        plan = planner.build_session_plan(
            subject=sub,
            session=ses,
            ref=ref_path,
            qmri_files=qmri_files,
            source_map=source_map,
            preproc_config=preproc_config,
        )

        session_plans.append(plan)

    Parallel(n_jobs=n_sessions, backend="loky")(
        delayed(run_session_parallel)(
            plan=plan,
            reg_config_template=config["registration"]["ref_to_template"],
            reg_config_qmri=config["registration"]["ref_to_qmri"],
            force=force,
            n_workers=n_workers,
        )
        for plan in session_plans
    )

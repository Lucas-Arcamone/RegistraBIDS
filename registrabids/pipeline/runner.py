import logging
from pathlib import Path
from registrabids.core.bids_index import BIDSIndex
from registrabids.core.resolver import ReferenceResolver, RegistrationPlanner
from registrabids.core.planner import SessionPlan, RegistrationJob, ApplyTransformJob
from registrabids.pipeline.registration import (
    run_registration, parse_registration_config
)
from registrabids.core.template import TemplateLoader
from registrabids.pipeline.preprocessing import run_preprocessing_plan
logger = logging.getLogger(__name__)


def _apply_transforms(job: ApplyTransformJob, transforms: list[str]) -> None:
    """
    Lance antsApplyTransforms pour amener la qmap dans l'espace template.
    transforms : liste ordonnée des fichiers de transform à chaîner,
                 du plus récent au plus ancien (convention ANTs).
    """
    import subprocess
    job.out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "antsApplyTransforms",
        "-d", "3",
        "-i", str(job.qmap),
        "-r", str(job.out_path),   # référence = espace cible (template)
        "-o", str(job.out_path),
        "--interpolation", "Linear",
    ]
    for t in transforms:
        cmd += ["-t", t]

    logger.info("Applying transforms → %s", job.out_path.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antsApplyTransforms a échoué pour {job.qmap.name} :\n"
            f"{result.stderr}"
        )

def run_session(
    plan: SessionPlan,
    reg_config_template: dict,   # bloc YAML registration.ref_to_template
    reg_config_qmri: dict,      # bloc YAML registration.ref_to_qmri 
    preproc_config=None,
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
        fixed = preprocessed.get("ref", job.fixed) if job.job_type == "ref_to_template" else preprocessed.get("ref", job.fixed)
        moving = preprocessed.get(job.source_key, job.moving)

        cfg = config_template if job.job_type == "ref_to_template" else config_qmri
        result = run_registration(
            fixed=fixed,
            moving=moving,
            out_prefix=job.out_prefix,
            config=cfg,
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
                app_job.source_key, app_job.qmap.name,
            )
            continue

        # ANTs order: from newest to oldest
        # T(ref→template) ∘ T(source→ref)
        transforms = [
            f"{prefix_template}1Warp.nii.gz",       # warp SyN
            f"{prefix_template}0GenericAffine.mat",  # affine ref→template
            f"{prefix_source}0GenericAffine.mat",    # affine source→ref
        ]
        _apply_transforms(app_job, transforms)

    logger.info(
        "Session sub-%s ses-%s done — %d qmaps in the template space.",
        plan.subject, plan.session, len(plan.apply_jobs),
    )

def run_pipeline(bids_root: str, config: dict) -> None:
    """
    Main entry point.
    config: a dictionary derived from the full YAML file.
    """
    index = BIDSIndex(bids_root)
    resolver = ReferenceResolver(index.layout)
    
    atlas = TemplateLoader.from_config(config["template"])
    template = atlas.template

    output_root = Path(bids_root) / "derivatives" / "registrabids"
    planner = RegistrationPlanner(index.layout, template, output_root)

    reference_map = resolver.extract_reference_map(config)
    grouped = index.get_qmri_maps_grouped(config)
    source_map = index.map_to_sources(config)

    for (sub, ses), qmri_files in grouped.items():
        ref_path = reference_map.get((sub, ses))
        if not ref_path:
            logger.warning("No reference for sub-%s ses-%s, session ignored.", sub, ses)
            continue

        if not ref_path:
            logger.error("Reference file not found in the layout : %s", ref_path[0])
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
        #print(plan.registration_jobs)
        try:
            run_session(
                plan=plan,
                reg_config_template=config["registration"]["ref_to_template"],
                reg_config_qmri=config["registration"]["ref_to_qmri"],
                preproc_config=preproc_config,
            )
        except RuntimeError as e:
            logger.error("Error sub-%s ses-%s : %s", sub, ses, e)
            continue
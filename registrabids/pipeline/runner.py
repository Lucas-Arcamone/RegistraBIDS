import logging
from pathlib import Path
from registrabids.core.bids_index import BIDSIndex
from registrabids.core.resolver import ReferenceResolver, RegistrationPlanner
from registrabids.core.planner import SessionPlan, RegistrationJob, ApplyTransformJob
from registrabids.pipeline.registration import (
    run_registration, parse_registration_config
)
from registrabids.core.template import TemplateLoader

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
    reg_config_qmri: dict,       # bloc YAML registration.ref_to_qmri
) -> None:
    """
    Exécute le plan complet pour une session :
      1. Recalages (ref→template, sources→ref)
      2. Application des transforms chaînées sur chaque qmap
    """
    config_template = parse_registration_config(reg_config_template)
    config_qmri = parse_registration_config(reg_config_qmri)

    # Stocke les prefixes de output par source_key pour récupérer les transforms
    transform_prefixes: dict[str, Path] = {}

    for job in plan.registration_jobs:
        cfg = config_template if job.job_type == "ref_to_template" else config_qmri
        result = run_registration(
            fixed=job.fixed,
            moving=job.moving,
            out_prefix=job.out_prefix,
            config=cfg,
        )
        transform_prefixes[job.source_key] = result["prefix"]
        logger.info("✓ %s terminé", job.source_key)

    # ── Application des transforms ───────────────────────────────────────
    prefix_template = transform_prefixes["ref"]

    for app_job in plan.apply_jobs:
        prefix_source = transform_prefixes.get(app_job.source_key)
        if prefix_source is None:
            logger.error(
                "Pas de transform trouvé pour source_key='%s', qmap ignorée : %s",
                app_job.source_key, app_job.qmap.name,
            )
            continue

        # Chaîne ANTs : du plus récent au plus ancien
        # T(ref→template) ∘ T(source→ref)
        transforms = [
            f"{prefix_template}1Warp.nii.gz",       # warp SyN
            f"{prefix_template}0GenericAffine.mat",  # affine ref→template
            f"{prefix_source}0GenericAffine.mat",    # affine source→ref
        ]
        _apply_transforms(app_job, transforms)

    logger.info(
        "Session sub-%s ses-%s terminée — %d qmaps dans l'espace template.",
        plan.subject, plan.session, len(plan.apply_jobs),
    )

def run_pipeline(bids_root: str, config: dict) -> None:
    """
    Point d'entrée principal.
    config : dict issu du YAML complet.
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
        ref_filename = reference_map.get((sub, ses))
        if not ref_filename:
            logger.warning("Pas de référence pour sub-%s ses-%s, session ignorée.", sub, ses)
            continue

        # Reconstruire le path complet depuis le layout
        ref_candidates = index.layout.get(
            subject=sub, session=ses,
            filename=ref_filename[0],
            extension=[".nii", ".nii.gz"],
        )
        if not ref_candidates:
            logger.error("Fichier ref introuvable dans le layout : %s", ref_filename[0])
            continue

        ref_path = Path(ref_candidates[0].path)

        plan = planner.build_session_plan(
            subject=sub,
            session=ses,
            ref=ref_path,
            qmri_files=qmri_files,
            source_map=source_map,
        )

        try:
            run_session(
                plan=plan,
                reg_config_template=config["registration"]["ref_to_template"],
                reg_config_qmri=config["registration"]["ref_to_qmri"],
            )
        except RuntimeError as e:
            logger.error("Erreur sub-%s ses-%s : %s", sub, ses, e)
            continue
from pathlib import Path

import logging
from registrabids.core.planner import (
    SessionPlan, RegistrationJob, ApplyTransformJob
)

logger = logging.getLogger(__name__)


class ReferenceResolver:
    def __init__(self, layout):
        self.layout = layout

    def extract_reference_map(self, config: dict) -> dict:
        ref_config = config["reference"]
        filter_cfg = config.get("filter", {})

        subjects = filter_cfg.get("subjects")
        sessions = filter_cfg.get("sessions")

        pairs = (
            self._pairs_from_filters(subjects, sessions)
            if (subjects is not None or sessions is not None)
            else self._discover_pairs()
        )

        if not pairs:
            raise ValueError(
                "No matches (subject, session) found. "
                "Check the BIDS structure or your filters. "
            )

        logger.info("%d pair(s) (sub, his) to be processed.", len(pairs))

        reference_map = {}
        errors = []

        for sub, ses in pairs:
            query = {
                "subject": sub,
                "session": ses,
                "extension": [".nii", ".nii.gz"],
            }
            query.update({k: v for k, v in ref_config.items() if v is not None})
            candidates = self.layout.get(**query)

            if not candidates:
                errors.append(f"sub-{sub} ses-{ses} with settings  : {ref_config}")
                logger.error("No references for sub-%s ses-%s", sub, ses)
                continue

            if len(candidates) > 1:
                logger.warning(
                    "Several candidates for sub-%s ses-%s — use of : %s",
                    sub, ses, candidates[0].filename,
                )

            reference_map[(sub, ses)] = Path(candidates[0].path)
            logger.debug("Reference sub-%s ses-%s : %s", sub, ses, candidates[0].filename)

        if errors:
            raise ValueError(
                f"{len(errors)} missing reference(s) :\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return reference_map

    def _discover_pairs(self) -> list[tuple[str, str]]:
        files = self.layout.get(extension=[".nii", ".nii.gz"])
        seen = set()
        for f in files:
            sub = f.entities.get("subject")
            ses = f.entities.get("session")
            if sub is not None and ses is not None:
                seen.add((sub, ses))
        return sorted(seen)

    def _pairs_from_filters(self, subjects, sessions) -> list[tuple[str, str]]:
        if isinstance(subjects, str):
            subjects = [subjects]
        if isinstance(sessions, str):
            sessions = [sessions]

        files = self.layout.get(
            subject=subjects,
            session=sessions,
            extension=[".nii", ".nii.gz"],
        )
        seen = set()
        for f in files:
            sub = f.entities.get("subject")
            ses = f.entities.get("session")
            if sub is not None and ses is not None:
                seen.add((sub, ses))

        if subjects:
            missing = set(subjects) - {s for s, _ in seen}
            if missing:
                logger.warning("Sujets absents du layout : %s", missing)
        if sessions:
            missing = set(sessions) - {s for _, s in seen}
            if missing:
                logger.warning("Sessions absentes du layout : %s", missing)

        return sorted(seen)
    
    
def _source_key(source_path: str) -> str:
    """
    Extrait un identifiant court depuis le path d'une source rawdata.
    Ex: '.../sub-M30_ses-..._acq-refMT_run-02_res-100iso_T1w.nii.gz'
        → 'acq-refMT_run-02_res-100iso_T1w'
    """
    name = Path(source_path).name
    # retire l'extension .nii.gz ou .nii
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    # retire le préfixe sub-XX_ses-YY_ pour garder uniquement les entités utiles
    parts = name.split("_")
    return "_".join(p for p in parts if not p.startswith(("sub-", "ses-")))


class RegistrationPlanner:
    def __init__(self, layout, template: Path, output_root: Path):
        self.layout = layout
        self.template = template
        self.output_root = output_root  # e.g. derivatives/registrabids/

    def build_session_plan(
        self,
        subject: str,
        session: str,
        ref: Path,
        qmri_files,           # liste de BIDSImageFile
        source_map: dict,     # {qmap_path: [source_path, ...]}
    ) -> SessionPlan:
        """
        Build the complete plan for a (sub, ses):
          - 1 job ref → template
          - N jobs source_i → ref  (deduplicated by source)
          - M apply jobs (one per qmap)
        """
        out_base = (
            self.output_root
            / f"sub-{subject}"
            / f"ses-{session}"
        )

        plan = SessionPlan(
            subject=subject,
            session=session,
            ref=ref,
            template=self.template,
        )

        # ── Job 1 : ref → template ──────────────────────────────────────
        plan.registration_jobs.append(
            RegistrationJob(
                fixed=self.template,
                moving=ref,
                out_prefix=out_base / "ref_to_template" / "ref_to_template",
                job_type="ref_to_template",
                source_key="ref",
            )
        )

        # ── Jobs source → ref (dédupliqués) ─────────────────────────────
        seen_sources: dict[str, Path] = {}   # source_key → source Path

        for bids_file in qmri_files:
            qmap_path = bids_file.path
            sources = source_map.get(qmap_path, [])

            if not sources:
                logger.warning(
                    "No source found for %s — qmap ignored.",
                    Path(qmap_path).name,
                )
                continue

            # On prend la première source (en pratique il n'y en a qu'une)
            source_path = Path(sources[0])
            key = _source_key(str(source_path))

            # avoid registration job already planned because several qmri can have the same source.
            if key not in seen_sources:
                seen_sources[key] = source_path
                plan.registration_jobs.append(
                    RegistrationJob(
                        fixed=ref,
                        moving=source_path,
                        out_prefix=out_base / f"source_to_ref_{key}" / key,
                        job_type="source_to_ref",
                        source_key=key,
                    )
                )

            # ── Apply job pour cette qmap ────────────────────────────────
            qmap_out_name = Path(qmap_path).name.replace(
                ".nii.gz", "_space-template.nii.gz"
            ).replace(".nii", "_space-template.nii.gz")

            plan.apply_jobs.append(
                ApplyTransformJob(
                    qmap=Path(qmap_path),
                    source_key=key,
                    out_path=out_base / "warped" / qmap_out_name,
                )
            )

        self._log_plan(plan)
        return plan

    def _log_plan(self, plan: SessionPlan) -> None:
        logger.info(
            "Plan sub-%s ses-%s : %d recalage(s), %d qmap(s) à appliquer",
            plan.subject, plan.session,
            len(plan.registration_jobs),
            len(plan.apply_jobs),
        )
        for job in plan.registration_jobs:
            logger.debug(
                "  [%s] %s → %s",
                job.job_type,
                job.moving.name,
                job.fixed.name,
            )
        for app in plan.apply_jobs:
            logger.debug(
                "  [apply] %s via source_key='%s'",
                app.qmap.name,
                app.source_key,
            )
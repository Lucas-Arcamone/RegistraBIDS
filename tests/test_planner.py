import pytest
from pathlib import Path
from unittest.mock import MagicMock
from registrabids.core.planner import SessionPlan, RegistrationJob, ApplyTransformJob
from registrabids.core.resolver import RegistrationPlanner


# ─────────────────────────────────────────
# Helpers — faux objets BIDS
# ─────────────────────────────────────────

def make_bids_file(path: str) -> MagicMock:
    """Simule un BIDSImageFile avec juste un .path."""
    f = MagicMock()
    f.path = path
    return f


# ─────────────────────────────────────────
# Fixture commune
# ─────────────────────────────────────────

@pytest.fixture
def planner(tmp_path):
    template = tmp_path / "ABAv3_template.nii.gz"
    template.touch()
    output_root = tmp_path / "derivatives" / "registrabids"
    return RegistrationPlanner(
        layout=None,           # non utilisé par build_session_plan
        template=template,
        output_root=output_root,
    )


@pytest.fixture
def base_inputs():
    """Inputs minimaux pour une session avec 2 sources distinctes."""
    ref = Path("/data/sub-M30/ses-01/anat/sub-M30_ses-01_acq-flash_T1w.nii.gz")

    qmri_files = [
        make_bids_file("/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTRmap.nii"),
        make_bids_file("/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTsat.nii"),
        make_bids_file("/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_acq-OGSE_AD.nii"),
    ]

    source_map = {
        "/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTRmap.nii":
            ["/data/sub-M30/ses-01/anat/sub-M30_ses-01_acq-refMT_run-02_T1w.nii.gz"],
        "/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTsat.nii":
            ["/data/sub-M30/ses-01/anat/sub-M30_ses-01_acq-refMT_run-02_T1w.nii.gz"],
        "/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_acq-OGSE_AD.nii":
            ["/data/sub-M30/ses-01/dwi/sub-M30_ses-01_acq-ogse_dwi.nii.gz"],
    }

    return ref, qmri_files, source_map


# ─────────────────────────────────────────
# Tests
# ─────────────────────────────────────────

class TestBuildSessionPlan:

    def test_retourne_un_session_plan(self, planner, base_inputs):
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        assert isinstance(plan, SessionPlan)
        assert plan.subject == "M30"
        assert plan.session == "01"
        assert plan.ref == ref
        assert plan.template == planner.template

    def test_toujours_un_job_ref_to_template(self, planner, base_inputs):
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        ref_to_template = [j for j in plan.registration_jobs if j.job_type == "ref_to_template"]
        assert len(ref_to_template) == 1
        assert ref_to_template[0].fixed == planner.template
        assert ref_to_template[0].moving == ref
        assert ref_to_template[0].source_key == "ref"

    def test_deduplication_des_sources(self, planner, base_inputs):
        """MTRmap et MTsat partagent la même source → 1 seul job source_to_ref."""
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        source_jobs = [j for j in plan.registration_jobs if j.job_type == "source_to_ref"]
        # 2 sources distinctes : acq-refMT et acq-ogse
        assert len(source_jobs) == 2

    def test_nombre_apply_jobs(self, planner, base_inputs):
        """Un apply job par qmap, même si les sources sont dédupliquées."""
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        assert len(plan.apply_jobs) == 3

    def test_source_key_coherent_entre_job_et_apply(self, planner, base_inputs):
        """Chaque apply_job.source_key doit correspondre à un registration_job."""
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        job_keys = {j.source_key for j in plan.registration_jobs}
        for app in plan.apply_jobs:
            assert app.source_key in job_keys, (
                f"source_key '{app.source_key}' dans apply_jobs "
                f"sans registration_job correspondant"
            )

    def test_output_paths_sous_out_base(self, planner, base_inputs):
        """Tous les outputs doivent être sous output_root/sub-XX/ses-YY/."""
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        expected_base = planner.output_root / "sub-M30" / "ses-01"
        for job in plan.registration_jobs:
            assert str(job.out_prefix).startswith(str(expected_base))
        for app in plan.apply_jobs:
            assert str(app.out_path).startswith(str(expected_base))

    def test_qmap_sans_source_ignoree(self, planner, tmp_path):
        """Une qmap sans source dans source_map ne génère pas d'apply_job."""
        ref = Path("/data/sub-M30/ses-01/anat/ref.nii.gz")
        qmri_files = [
            make_bids_file("/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTRmap.nii"),
            make_bids_file("/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_orphan.nii"),
        ]
        source_map = {
            "/data/derivatives/qmri/sub-M30/ses-01/sub-M30_ses-01_MTRmap.nii":
                ["/data/sub-M30/ses-01/anat/sub-M30_ses-01_acq-refMT_T1w.nii.gz"],
            # orphan.nii absent du source_map
        }
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        assert len(plan.apply_jobs) == 1
        assert plan.apply_jobs[0].qmap.name == "sub-M30_ses-01_MTRmap.nii"

    def test_output_warped_suffix(self, planner, base_inputs):
        """Les qmaps warpées doivent avoir le suffix _space-template."""
        ref, qmri_files, source_map = base_inputs
        plan = planner.build_session_plan("M30", "01", ref, qmri_files, source_map)
        for app in plan.apply_jobs:
            assert "_space-template" in app.out_path.name

    def test_session_sans_qmri(self, planner):
        """Session sans aucune qmap → plan vide mais valide."""
        ref = Path("/data/sub-M30/ses-01/anat/ref.nii.gz")
        plan = planner.build_session_plan("M30", "01", ref, [], {})
        assert len(plan.apply_jobs) == 0
        # Le job ref→template existe quand même
        assert len(plan.registration_jobs) == 1
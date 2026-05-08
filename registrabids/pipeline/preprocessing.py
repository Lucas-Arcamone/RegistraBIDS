from __future__ import annotations
import logging
import shutil
import tempfile
from registrabids.core.planner import (VolumeExtractionJob, DenoisingJob, N4Job, PreprocessingPlan)
from pathlib import Path

import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)



# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────

def _save_nifti_safe(img: nib.Nifti1Image, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, dest)
    logger.debug("Save → %s", dest.name)

# ─────────────────────────────────────────
# Parsers — config → dataclasses
# ─────────────────────────────────────────

def _match_rule(entities: dict, rule_match: dict) -> bool:
    """
    Vérifie si les entités BIDS d'un fichier matchent une règle.
    Un match vide {} matche tout (fallback).
    """
    return all(entities.get(k) == v for k, v in rule_match.items())


def _resolve_strategy(
    entities: dict,
    rules: list[dict],
) -> tuple[str, dict]:
    """
    Returns (strategy, params) for a given file.
    The first rule that matches wins.
    Fallback: first_volume if no rule matches.
    """
    for rule in rules:
        if _match_rule(entities, rule.get("match", {})):
            return rule["strategy"], rule.get("params", {})
    logger.warning(
        "No volume_extraction rule matches %s — fallback : first_volume",
        entities,
    )
    return "first_volume", {}

def _clean_stem(path: Path) -> str:
    name = path.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem

def build_preprocessing_plan(
    source_key: str,
    file_path: Path,
    entities: dict,          # entités BIDS du fichier (suffix, acquisition, ...)
    preproc_config: dict,
    out_base: Path,
) -> PreprocessingPlan:
    """
    Builds the preprocessing plan for a source or reference file.
    """
    save_cfg = (preproc_config or {}).get("save_intermediates", {})
    extraction_cfg = (preproc_config or {}).get("volume_extraction", {})
    n4_cfg = (preproc_config or {}).get("n4", {})
    denoising_cfg = (preproc_config or {}).get("denoising", {})

    suffix = entities.get("suffix", "")
    preproc_dir = out_base / "preproc" / source_key
    preproc_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    current_path = file_path
    stem = _clean_stem(file_path)

    # ── Step 1: 3D extraction, if necessary ───────────────────────────
    img = nib.load(file_path)
    is_4d = img.ndim == 4 and img.shape[3] > 1

    if is_4d:
        rules = extraction_cfg.get("rules", [])
        strategy, params = _resolve_strategy(entities, rules)
        save_extraction = save_cfg.get("extraction", False)

        ext_out = (
            preproc_dir / f"{stem}_extracted.nii.gz"
            if save_extraction
            else _temp_nifti("_extracted.nii.gz")
        )

        job = VolumeExtractionJob(
            input_path=current_path,
            output_path=ext_out,
            strategy=strategy,
            params=params,
            save=save_extraction,
        )
        jobs.append(job)
        current_path = ext_out

    # ── Step 2: denoising ────────────────────────────────────────────
    if denoising_cfg.get("enabled", False):
        save_denoising = save_cfg.get("denoising", False)

        den_out = (
            preproc_dir / f"{stem}_denoised.nii.gz"
            if save_denoising
            else _temp_nifti("_denoised.nii.gz")
        )

        job = DenoisingJob(
            input_path=current_path,
            output_path=den_out,
            method=denoising_cfg.get("method", "NLMF"),
            noise_model=denoising_cfg.get("noise_model", "rician"),
            patch_radius=denoising_cfg.get("patch_radius", 1),
            search_radius=denoising_cfg.get("search_radius", 3),
            save=save_denoising,
        )
        jobs.append(job)
        current_path = den_out

    # ── Step 3: N4 ────────────────────────────────────────────────────
    skip_suffixes = n4_cfg.get("skip_suffixes", [])
    n4_enabled = n4_cfg.get("enabled", False) and suffix not in skip_suffixes

    if n4_enabled:
        save_n4 = save_cfg.get("n4", False)

        n4_out = (
            preproc_dir / f"{stem}_N4.nii.gz"
            if save_n4
            else _temp_nifti("_N4.nii.gz")
        )
        
        job = N4Job(
            input_path=current_path,
            output_path=n4_out,
            shrink_factor=n4_cfg.get("shrink_factor", 4),
            n_iterations=n4_cfg.get("n_iterations", [50, 50, 30, 20]),
            convergence_threshold=n4_cfg.get("convergence_threshold", 0.001),
            save=save_n4,
        )
        jobs.append(job)
        current_path = n4_out

    # If no job: the file is already ready (3D, no N4, no denoising)
    return PreprocessingPlan(
        source_key=source_key,
        original_path=file_path,
        jobs=jobs,
        final_path=current_path,
    )


# ─────────────────────────────────────────
# Executors — one per Job type
# ─────────────────────────────────────────

def run_volume_extraction(job: VolumeExtractionJob) -> None:
    img = nib.load(job.input_path)
    data = np.asarray(img.dataobj)

    strategy = job.strategy

    if strategy == "first_volume":
        vol = data[..., 0]

    elif strategy == "mean":
        vol = np.mean(data, axis=3)

    elif strategy == "geometric_mean_shell":
        vol = _geometric_mean_shell(job.input_path, data, job.params)

    elif strategy == "weighted_mean_echo":
        vol = _weighted_mean_echo(job.input_path, data)

    else:
        raise ValueError(
            f"Unknown extraction strategy : '{strategy}'. "
            f"Accepted values : first_volume, mean, geometric_mean_shell, "
            f"weighted_mean_echo."
        )

    out_img = nib.Nifti1Image(vol.astype(np.float32), img.affine, img.header)
    _save_nifti_safe(out_img, job.output_path)
    logger.info("Extraction [%s] → %s", strategy, job.output_path.name)


def _geometric_mean_shell(
    file_path: Path, data: np.ndarray, params: dict
) -> np.ndarray:
    """Geometric mean of the volumes if bval >= 50 for a target DWI shell
    Arithmetic mean if bval < 50."""
    bval_path = file_path.with_suffix("").with_suffix(".bval")
    if not bval_path.exists():
        # Cherche dans le même dossier avec le même stem
        bval_path = file_path.parent / (file_path.name.split(".")[0] + ".bval")
    if not bval_path.exists():
        raise FileNotFoundError(
            f".bval file not found for {file_path.name}. "
            f"Searched : {bval_path}"
        )

    bvals = np.loadtxt(bval_path)
    target = params.get("target_bval", 0)
    tolerance = params.get("bval_tolerance", 50)
    indices = np.where(np.abs(bvals - target) <= tolerance)[0]

    if len(indices) == 0:
        raise ValueError(
            f"No results found for b={target} ± {tolerance} "
            f"in {bval_path.name}. "
            f"Available values : {np.unique(bvals).tolist()}"
        )

    shell = data[..., indices].astype(np.float64)
    # If bval = 0 automatically realise Mean extraction
    # else perform geometric mean on a shell (better contrast)
    if target < 50:
        vol = np.mean(shell, axis=3)
        logger.debug(
            "Shell b=%d -- Mean strategy: %d selected volume(s) (indices %s)",
            target, len(indices), indices.tolist(),
        )
    else:
        shell = np.clip(shell, 1e-8, None)
        vol = np.exp(np.mean(np.log(shell), axis=3))
        logger.debug(
            "Shell b=%d : %d selected volume(s) (indices %s)",
            target, len(indices), indices.tolist(),
        )
    return vol

#Designed for MEGRE but may be applied to MESE ?  
def _weighted_mean_echo(file_path: Path, data: np.ndarray) -> np.ndarray:
    """
    Weighted multi-echo average.
    Weight ∝ TE × exp(-TE / T2*_estimate).
    TE values read from the JSON sidecar (list or single value).
    """
    import json

    json_path = file_path.with_suffix("").with_suffix(".json")
    if not json_path.exists():
        json_path = file_path.parent / (file_path.name.split(".")[0] + ".json")
    if not json_path.exists():
        logger.warning(
            "JSON sidecar not found for %s — fallback mean", file_path.name
        )
        return np.mean(data, axis=3)

    with open(json_path) as f:
        meta = json.load(f)

    echo_times = meta.get("EchoTime", None)
    if echo_times is None:
        logger.warning("EchoTime is not available on the sidecar — fallback mean")
        return np.mean(data, axis=3)

    if isinstance(echo_times, (int, float)):
        echo_times = [echo_times]

    echo_times = np.array(echo_times, dtype=np.float64)
    n_echoes = data.shape[3]

    if len(echo_times) != n_echoes:
        logger.warning(
            "Number of EchoTime (%d) ≠ number of volumes (%d) — fallback mean",
            len(echo_times), n_echoes,
        )
        return np.mean(data, axis=3)

    # Simple estimate of T2*: 40% of T_E max
    t2star_estimate = 0.4 * echo_times.max()
    weights = echo_times * np.exp(-echo_times / t2star_estimate)
    weights /= weights.sum()

    logger.debug("Multi-echo weighting : %s", np.round(weights, 3).tolist())
    vol = np.tensordot(data.astype(np.float64), weights, axes=([3], [0]))
    return vol


def run_denoising(job: DenoisingJob) -> None:
    img = nib.load(job.input_path)
    data = np.asarray(img.dataobj, dtype=np.float64)
    
    if img.ndim == 4:
        raise RuntimeError(
            f"run_denoising a reçu un volume 4D : {job.input_path.name}."
        )
    
    # Garantit que le dossier de destination existe
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(
        "Output path : %s (parent exists=%s)",
        job.output_path,
        job.output_path.parent.exists(),
    )
    
    data = np.asarray(img.dataobj, dtype=np.float32)

    if job.method == "NLMF":
        from dipy.denoise.nlmeans import nlmeans
        from dipy.denoise.noise_estimate import estimate_sigma
        from dipy.denoise.localpca import mppca
        sigma = estimate_sigma(data, N=1)
        denoised = nlmeans(
            data,
            sigma=sigma,
            patch_radius=job.patch_radius,
            block_radius=job.search_radius,
            rician=job.noise_model == "rician",
        )
    elif job.method == "MPPCA":
        from dipy.denoise.localpca import mppca
        denoised, _ = mppca(data, patch_radius=job.patch_radius, return_sigma=True)
    else:
        raise ValueError(
            f"Méthode de débruitage inconnue : '{job.method}'. "
            f"Valeurs acceptées : NLMF, MPPCA."
        )

    out_img = nib.Nifti1Image(denoised.astype(np.float32), img.affine, img.header)
    _save_nifti_safe(out_img, job.output_path)
    logger.info("Débruitage [%s/%s] → %s", job.method, job.noise_model, job.output_path.name)


def run_n4(job: N4Job) -> None:
    import subprocess

    job.output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "N4BiasFieldCorrection",
        "-d", "3",
        "-i", str(job.input_path),
        "-o", str(job.output_path),
        "-s", str(job.shrink_factor),
        "-c", f"[{'x'.join(str(i) for i in job.n_iterations)},{job.convergence_threshold}]",
    ]

    logger.info("N4BiasFieldCorrection : %s", job.input_path.name)
    logger.debug("Commande : %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"N4BiasFieldCorrection failed for {job.input_path.name} :\n"
            f"{result.stderr}"
        )

    logger.info("N4 → %s", job.output_path.name)

def _temp_nifti(suffix: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="registrabids_", dir="/tmp"))
    out = tmp_dir / f"tmp{suffix}"
    logger.debug("Fichier temporaire prévu : %s (dir exists=%s)", out, tmp_dir.exists())
    return out

def run_preprocessing_plan(plan: PreprocessingPlan) -> Path:
    """
    Exécute tous les jobs d'un PreprocessingPlan dans l'ordre.
    Retourne le path du fichier préprocessé final.
    """
    if not plan.jobs:
        logger.debug(
            "[%s] Aucun preprocessing nécessaire — fichier utilisé tel quel.",
            plan.source_key,
        )
        return plan.original_path

    logger.info(
        "[%s] Preprocessing : %d étape(s)",
        plan.source_key, len(plan.jobs),
    )

    for job in plan.jobs:
        if isinstance(job, VolumeExtractionJob):
            run_volume_extraction(job)
        elif isinstance(job, DenoisingJob):
            run_denoising(job)
        elif isinstance(job, N4Job):
            run_n4(job)
        else:
            raise TypeError(f"Job de preprocessing inconnu : {type(job)}")

    return plan.final_path
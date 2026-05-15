# RegistraBIDS

**RegistraBIDS** automates the registration of quantitative MRI (qMRI) images in a common atlas space based on acquired raw data to optimize registration quality, starting from a dataset that complies with the BIDS standard. It uses ANTs under the hoods and requires no manual path management: everything is controlled by a single YAML configuration file.

The pipeline handles the full registration workflow:
- **Preprocessing** — automatic 4D→3D volume extraction with different strategy (e.g. mean, first volume, dwi geometric mean, etc.), optional denoising (NLMF, MPPCA) and bias field correction (N4), all configurable per acquisition type
- **Registration** — SyN registration of the reference image to the atlas template, followed by rigid/affine registration of each source image to the reference
- **Transform application** — qMRI maps are brought into atlas space by chaining transforms, preserving their native resolution
- **Parallelism** — sessions and intra-session jobs (preprocessing, registration, apply transforms) run in parallel via joblib, with automatic CPU allocation
- **Resume support** — completed steps are automatically skipped on re-run; use `--force` to reprocess

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset structure](#dataset-structure)
- [Configuration file](#configuration-file)
  - [Minimal example](#minimal-example)
  - [Processing only specific subjects/sessions](#processing-only-specific-subjectssessions)
  - [Perform preprocesssing](#perform-preprocesssing)
  - [Parallelism management](#parallelism-management)
  - [Configuration reference](#configuration-file)
- [Running the pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Available atlases](#available-atlases)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- Python ≥ 3.11
- [pixi](https://prefix.dev/) for dependency management
- ANTs (must be available in your `$PATH`)

Verify ANTs is accessible:

```bash
which antsRegistration
antsRegistration --version
```

---

## Installation

```bash
git clone https://github.com/Lucas-Arcamone/RegistraBIDS.git
cd RegistraBIDS
pixi install
```

This repository use pre-commit:
```bash
pixi run pre-commit install
```

---

## Dataset structure

RegistraBIDS expects a BIDS dataset with the following layout:

```
my_dataset/
├── sub-01/
│   └── ses-01/
│       ├── anat/
|           ├── sub-01_ses-..._acq-flash_run-02_T1w.nii.gz   ← reference
│       │   └── ... ← other rawdata
│       └── dwi/
|           ├── sub-01_ses-..._acq-ogse_dwi.nii.gz
│           └── ...
└── derivatives/
    └── qmri/
        └── sub-01/
            └── ses-01/
                ├── sub-01_ses-..._MTRmap.nii          ← qMRI maps
                ├── sub-01_ses-..._MTRmap.json          ← must contain "Sources"
                ├── sub-01_ses-..._MTsat.nii
                ├── sub-01_ses-..._MTsat.json
                ├── sub-01_ses-..._acq-OGSE_AD.nii
                ├── sub-01_ses-..._acq-OGSE_AD.json
                └── ...
```

**Important:** every qMRI map must have an associated JSON sidecar with a `Sources` field pointing to the rawdata file used to generate it:

```json
{
  "Sources": [
    "/absolute/path/to/sub-01/ses-.../anat/sub-01_ses-..._acq-refMT_run-02_T1w.nii.gz"
  ],
  "ModalityType": "MTRmap"
}
```

This is what allows RegistraBIDS to chain transforms correctly — the qMRI maps are never registered directly, only the source anatomical images are.

---

## Configuration file

Create a YAML file (e.g. `config.yaml`) in your dataset's `derivatives/registrabids/` folder.

### Minimal example

```yaml
reference:
  suffix: T1w
  acquisition: flash
  run: "02"

template:
  atlas: ABAv3

registration:
  ref_to_template:
    stages:
      - metric: MI
        metric_params: {weight: 1, bins: 32, sampling: Regular, sampling_rate: 0.25}
        transform: Rigid
        transform_params: [1]
        convergence: [200, 100, 50]
        shrink_factors: [4, 2, 1]
        smoothing_sigmas: [4, 2, 1]
      - metric: MI
        metric_params: {weight: 1, bins: 32, sampling: Regular, sampling_rate: 0.25}
        transform: Affine
        transform_params: [1]
        convergence: [200, 100, 50]
        shrink_factors: [4, 2, 1]
        smoothing_sigmas: [4, 2, 1]
      - metric: Mattes
        metric_params: {weight: 1, bins: 32, sampling: Regular, sampling_rate: 0.6}
        transform: SyN
        transform_params: [0.5, 3, 1]
        convergence: [200, 100, 100]
        shrink_factors: [4, 2, 2]
        smoothing_sigmas: [4, 2, 2]
        use_histogram_matching: true

  ref_to_qmri:
    stages:
      - metric: MI
        metric_params: {weight: 1, bins: 32, sampling: Regular, sampling_rate: 0.25}
        transform: Rigid
        transform_params: [1]
        convergence: [200, 100, 50]
        shrink_factors: [4, 2, 1]
        smoothing_sigmas: [4, 2, 1]
      - metric: MI
        metric_params: {weight: 1, bins: 32, sampling: Regular, sampling_rate: 0.25}
        transform: Affine
        transform_params: [1]
        convergence: [200, 100, 50]
        shrink_factors: [4, 2, 1]
        smoothing_sigmas: [4, 2, 1]
```

### Processing only specific subjects/sessions

Add an optional `filter` block:

```yaml
filter:
  subjects: ["01", "02"]
  sessions: ["01"]
```

If `filter` is absent, all subjects and sessions in the dataset are processed.

### Perform preprocesssing
You can choose to apply preprocessing algorithm in order to improve registrations.
If `preprocessing` is absent, only the extraction is performed using the `first_volume` rule.
With the `save_intermediates` extractor you can choose to save intermediate (use bool: true or false) files from extraction, denoiser and n4 process.

```yaml
preprocessing:
  save_intermediates:
    extraction: true
    denoising: true
    n4: true

  volume_extraction:
    rules:
      - match: {suffix: T1w, acquisition: flash}
        strategy: mean

      - match: {suffix: T1w, acquisition: refMT}
        strategy: mean

      - match: {suffix: dwi, acquisition: ogse}
        strategy: geometric_mean_shell
        params:
          target_bval: 0

      - match: {suffix: bold}
        strategy: weighted_mean_echo

      # default fallback if no rule matches → first_volume
      # can also be declared explicitly:
      # - match: {}
      #   strategy: first_volume

  n4:
    enabled: true
    skip_suffixes: [dwi]        # ← N4 never applied on diffusion MRI
    shrink_factor: 4
    n_iterations: [50, 50, 30, 20]
    convergence_threshold: 0.001

  denoising:
    enabled: true
    method: NLMF                # NLMF | MPPCA
    noise_model: rician         # rician | gaussian
    patch_radius: 1
    search_radius: 3
```

#### 4D to 3D strategies
The `volume_extraction` extractor consist of the strategy to go from a 4D volume to a 3D volume. You can control to apply the extraction strategy according to BIDS sidecars (e.g. suffix: MEGRE, echo: 01). Here are described the different strategies implemented  :
 - `mean`: extract the mean volume;
 - `geometric_mean_shell` with `target_val` paramater: If `target_val` is choosen to be 0, then a mean strategy is automatically applied otherwise concatenate the volume by applying geometric mean;
 - `weighted_mean_echo`: For multi-echoes acquisition, extract the weighted mean echo according to a naïve mono-exponential model approch. The first echoes are more weighted than the last ones maximizing the quality.
 - `first_volume`: Extract the first volume.

#### Bias field correction
You can choose to automatically applied N4 algorithm. This part use the `N4BiasFieldCorrection` command line, make shure you have access to this function in your computer.

#### Denoising
Finally, you can choose to correct your data from noise using either MPPCA or NLMF strategies. Both denoising strategy use DIPY implementation. You can use to correct from Rician or Gaussian noise when using NLMF.

### Parallelism management

This pipeline automatically uses available resources to run processes in parallel. You can control parallelism via the configuration file:

```yaml
parallelism:
  n_sessions: 2  # number of sessions running in parallel
  n_workers: 4   # number of parallel tasks within each session
```

ANTs threads are allocated automatically: `available_CPUs / (n_sessions × n_workers)`.

For example, with 32 CPUs, `n_sessions: 2` and `n_workers: 4`:
- 2 × 4 = 8 ANTs processes run simultaneously
- Each ANTs process receives 32 / 8 = **4 threads**

To disable parallelism entirely (useful for debugging):

```yaml
parallelism:
  disabled: true
```
If no `parallelism` block is specified in the configuration file, the pipeline
automatically detects the available CPUs and defaults to:

- `n_sessions = 1` — one session at a time
- `n_workers = available_CPUs` — all CPUs allocated to parallel tasks within the session

Each ANTs process then receives `available_CPUs / n_workers = 1` thread.
While this default maximizes intra-session parallelism, it is not optimal for
the SyN registration step between the reference and the template, which is
computationally intensive and benefits from multi-threading.

For better performance, we recommend explicitly setting `n_workers` to the number
of independent registration jobs (typically 2–4) so that each ANTs process
receives more threads:

```yaml
parallelism:
  n_sessions: 1
  n_workers: 3   # ref→template + source_A→ref + source_B→ref
```

With 32 CPUs and `n_workers: 3`, each ANTs process receives `32 / 3 ≈ 10` threads,
significantly accelerating the SyN step.

### Configuration reference

| Key | Required | Description |
|---|---|---|
| `reference` | ✅ | BIDS entities identifying the reference image (T1w, acq, run, etc.) |
| `template.atlas` | ✅ | Name of the atlas folder inside `resources/` |
| `registration.ref_to_template` | ✅ | ANTs stages for reference → atlas registration (typically Rigid + Affine + SyN) |
| `registration.ref_to_qmri` | ✅ | ANTs stages for source → reference registration (typically Rigid + Affine) |
| `filter.subjects` | ❌ | List of subject IDs to process. If absent: all subjects |
| `filter.sessions` | ❌ | List of session IDs to process. If absent: all sessions |
| `preprocessing.save_intermediaites` | ❌ | Save intermediates files. If absent, the intermedates files are not saved |
| `preprocessing.volume_extraction` | ❌ | Apply the extraction strategy to files that match BIDS sidecars. Available strategies: `mean`, `geometric_mean_shell`, `weighted_mean_echo` and `first_volume`. If absent, use `first_volume` strategy. |
| `preprocessing.n4` | ❌ | Apply N4 bias field correction (ANTS) |
| `preprocessing.denoising` | ❌ | Apply choosen denoising algrorithm. Available algorithms: MPPCA and NLMF (DIPY) |
| `parallelism` | ❌ | Controls parallel execution. `n_sessions`: number of sessions running simultaneously. `n_workers`: number of parallel tasks within each session. If absent, defaults to `n_sessions=1` and `n_workers=available_CPUs` (see **Parallelism management**) |
---


## Running the pipeline

```bash
pixi run python -m registrabids.cli /path/to/my_dataset \
  --config /path/to/my_dataset/derivatives/registrabids/config.yaml
```

Log level can be controlled with `--log-level`. The output directory can be controlled with `--output-dir`. You can control to force re-execution of all steps, ignoring existing outputs by using `--force`.

```bash
pixi run python -m registrabids.cli /path/to/my_dataset \
  --config /path/to/config.yaml \
  --log-level DEBUG
  --output-dir /path/to/a/different/directory/
  --force
```

---

## Outputs

Results are written to `derivatives/registrabids/` following this structure:

```
derivatives/registrabids/
└── sub-01/
    └── ses-01/
        ├── preproc/
        │   ├── ref
        │   │   ├── ref_denoised.nii.gz
        │   │   └── ref_N4.nii.gz
        │   └── ...
        ├── ref_to_template/
        │   ├── ref_to_template_0GenericAffine.mat
        │   ├── ref_to_template_1Warp.nii.gz
        │   ├── ref_to_template_1InverseWarp.nii.gz
        │   ├── ref_to_template_warped.nii.gz       ← reference in atlas space
        │   ├── ref_to_template_inv_warped.nii.gz   ← atlas in reference space
        │   └── ref_to_template.log
        ├── source_to_ref_acq-refMT_run-02_T1w/
        │   ├── acq-refMT_run-02_T1w_0GenericAffine.mat
        │   ├── acq-refMT_run-02_T1w_warped.nii.gz
        │   └── acq-refMT_run-02_T1w.log
        ├── source_to_ref_acq-ogse_dwi/
        │   ├── acq-ogse_dwi_0GenericAffine.mat
        │   ├── acq-ogse_dwi_warped.nii.gz
        │   ├── acq-ogse_dwi_inv_warped.nii.gz
        │   └── acq-ogse_dwi.log
        └── warped/
            ├── sub-01_ses-..._MTRmap_space-template.nii.gz
            ├── sub-01_ses-..._MTsat_space-template.nii.gz
            └── sub-01_ses-..._acq-OGSE_AD_space-template.nii.gz
```

The `warped/` folder contains the final qMRI maps in atlas space, ready for analysis.

---

## Available atlases

| Atlas | Species | Description |
|---|---|---|
| `ABAv3` | Mouse | Allen Brain Atlas v3 — 10 µm template with parcellations |

To list atlases available in your installation:

```python
from registrabids.core.template import TemplateLoader
print(TemplateLoader().available_atlases())
```

---

## Troubleshooting

**`ValueError: 'acq' is not a recognized entity.` when using config file.**
The available entities may differ from BIDS sidecars. For example `acq` is not recognize so you need to use `acquisition`. To find out which entities you can choose, you can use the following code:
```python
from bids import BIDSLayout
layout = BIDSLayout('/path/to/dataset/', validate = False)
layout.entities.keys()
```

**`No reference found for sub-XX ses-YY`**
The BIDS entities in your `reference` block do not match any file for that subject/session. Check the exact filename with:
```bash
ls /path/to/dataset/sub-XX/ses-YY/anat/
```
and align the `acq`, `run`, `res`, `suffix` fields in your config accordingly.

**`No source found for <qmap> — qmap ignored`**
The JSON sidecar for that qMRI map is missing or has an empty `Sources` field. The map will be skipped. Add the correct `Sources` entry to the sidecar.

**ANTs registration fails (non-zero return code)**
Check the `.log` file next to the transform outputs. Each registration job writes a detailed log at `<out_prefix>.log`.

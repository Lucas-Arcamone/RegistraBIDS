# RegistraBIDS — Developer & Contributor Guide

---

## Table of contents

- [Architecture overview](#architecture-overview)
- [Module reference](#module-reference)
  - [core/bids_index.py](#corebids_indexpy)
  - [core/resolver.py](#coreResolverpy)
  - [core/planner.py](#coreplannerpy)
  - [core/template.py](#coretemplatepy)
  - [pipeline/registration.py](#pipelineregistrationpy)
  - [pipeline/runner.py](#pipelinerunnerpy)
- [Data flow](#data-flow)
- [Configuration contract](#configuration-contract)
- [Adding a new atlas](#adding-a-new-atlas)
- [Adding a new registration stage type](#adding-a-new-registration-stage-type)
- [Testing](#testing)
- [Logging conventions](#logging-conventions)
- [Dependency management with pixi](#dependency-management-with-pixi)
- [Contributing](#contributing)

---

## Architecture overview

```
registrabids/
├── resources/
│   └── ABAv3/                        # Atlas files (shipped with the repo)
├── core/
│   ├── bids_index.py                 # BIDS layout wrapper
│   ├── resolver.py                   # Reference resolution + RegistrationPlanner
│   ├── planner.py                    # Dataclasses: SessionPlan, RegistrationJob, ApplyTransformJob
│   └── template.py                   # Atlas loader
├── pipeline/
│   ├── registration.py               # ANTs command builder + runner
│   └── runner.py                     # Orchestrator: run_pipeline, run_session
├── __init__.py
└── cli.py                            # Click entry point
```

**Design principles:**

- The YAML config is parsed **once** in `run_pipeline`. Every downstream component receives an already-parsed `dict` — no module re-reads the file.
- `subjects`/`sessions` filters are never passed as function arguments. They are read from `config["filter"]` inside each component.
- ANTs is called via `subprocess` (not ANTsPy) to keep the dependency surface minimal and allow cluster deployment where only the ANTs binary is available.
- Registration jobs are **planned before execution**: `RegistrationPlanner.build_session_plan` returns a `SessionPlan` dataclass that can be inspected, logged, or serialized before any ANTs call is made.

---

## Module reference

### `core/bids_index.py`

**`BIDSIndex(root: str)`**

Wraps `pybids.BIDSLayout` with `derivatives=["derivatives/qmri"]` and `validate=False`.

| Method | Signature | Description |
|---|---|---|
| `get_qmri_maps` | `(config, suffixes=None) → list[BIDSImageFile]` | Returns all qMRI maps filtered by `config["filter"]`. Excludes files without `subject` or `session` entities. |
| `get_qmri_maps_grouped` | `(config) → defaultdict[(sub,ses), list]` | Groups qMRI maps by `(subject, session)` tuple. |
| `map_to_sources` | `(config) → dict[path, list[path]]` | Reads each qmap's JSON sidecar and returns `{qmap_path: [source_path, ...]}`. |

**Key invariant:** all three methods call `get_qmri_maps(config)` internally, so the `filter` block is applied consistently across all three.

---

### `core/resolver.py`

#### `ReferenceResolver(layout)`

Resolves the anatomical reference file for each `(subject, session)` pair.

**`extract_reference_map(config: dict) → dict[(sub, ses), Path]`**

1. Reads `config["reference"]` for BIDS entity constraints (suffix, acq, run, res…).
2. Reads `config.get("filter", {})` for subject/session constraints.
3. Calls `_discover_pairs()` if no filter, `_pairs_from_filters()` otherwise.
4. For each pair, queries the layout and returns `{(sub, ses): Path}` with the absolute path to the reference file.

Errors are **collected** (not raised immediately) so the full list of missing references is reported in one pass.

#### `RegistrationPlanner(layout, template: Path, output_root: Path)`

Builds the registration plan for a session without executing anything.

**`build_session_plan(subject, session, ref, qmri_files, source_map) → SessionPlan`**

Produces:
- 1 `RegistrationJob` of type `ref_to_template` (fixed=template, moving=ref)
- N deduplicated `RegistrationJob` of type `source_to_ref` (one per unique source)
- M `ApplyTransformJob` (one per qmap, referencing the correct `source_key`)

Source deduplication uses `_source_key()` which strips `sub-XX_ses-YY_` prefixes from the source filename, keeping only discriminating entities (acq, run, res, suffix).

---

### `core/planner.py`

Pure dataclasses — no logic, no I/O.

```python
@dataclass
class RegistrationJob:
    fixed: Path
    moving: Path
    out_prefix: Path
    job_type: str        # "ref_to_template" | "source_to_ref"
    source_key: str      # short identifier, used to match ApplyTransformJob

@dataclass
class ApplyTransformJob:
    qmap: Path
    source_key: str      # must match a RegistrationJob.source_key
    out_path: Path

@dataclass
class SessionPlan:
    subject: str
    session: str
    ref: Path
    template: Path
    registration_jobs: list[RegistrationJob]
    apply_jobs: list[ApplyTransformJob]
```

**Invariant:** every `ApplyTransformJob.source_key` must correspond to exactly one `RegistrationJob.source_key` in the same `SessionPlan`. This is enforced at runtime in `run_session`.

---

### `core/template.py`

**`TemplateLoader(resources_dir=None)`**

Resolves atlas files from `registrabids/resources/`. The default `resources_dir` is computed relative to `__file__`, making the atlas path independent of the working directory.

**`load(atlas_name: str) → AtlasFiles`**

Scans `resources/<atlas_name>/` and resolves:

| Attribute | Matched suffixes |
|---|---|
| `template` | `_template.nii.gz`, `.nii.gz` (fallback) |
| `mask` | `_mask.nii.gz` |
| `annotation` | `_annotation.nii.gz`, `_dseg.nii.gz` |
| `extra` | all other `.nii.gz` files |

**`from_config(config: dict) → AtlasFiles`** — convenience classmethod that reads `config["template"]["atlas"]`.

---

### `pipeline/registration.py`

Builds and executes `antsRegistration` commands from `RegistrationConfig` dataclasses.

#### Dataclasses

```python
@dataclass
class MetricConfig:
    name: str            # MI, Mattes, CC, MeanSquares
    weight: float
    bins: int
    sampling: str        # Regular, Random, None
    sampling_rate: float

@dataclass
class StageConfig:
    metric: MetricConfig
    transform: str       # Rigid, Affine, SyN, BSplineSyN...
    transform_params: list[float]
    convergence: list[int]
    shrink_factors: list[int]
    smoothing_sigmas: list[float]
    use_histogram_matching: bool
    # __post_init__ validates that convergence/shrink/smooth have equal length

@dataclass
class RegistrationConfig:
    stages: list[StageConfig]
    init_transform: Optional[str]
    verbose: bool
    dimensionality: int
```

#### Key functions

**`parse_registration_config(reg_cfg: dict) → RegistrationConfig`**

Parses a YAML registration block (either `ref_to_template` or `ref_to_qmri`) into validated dataclasses. Raises `ValueError` if `convergence`, `shrink_factors` and `smoothing_sigmas` have mismatched lengths.

**`run_registration(fixed, moving, out_prefix, config, log_file=None) → dict`**

Builds the command, runs it via `subprocess`, verifies the output is a valid gzip file, and returns `{warped, inv_warped, prefix, log}`.

**Important:** the output prefix passed to ANTs ends with `_` (e.g. `ref_to_template_`) so that ANTs correctly infers compression format from the explicit `.nii.gz` extensions of the warped output paths.

**`_verify_nifti(path)`** — reads the first 4 bytes of the output file with `gzip.open` and raises `RuntimeError` if the file is not a valid gzip, providing a clear error with the log path before any downstream code tries to load the file.

---

### `pipeline/runner.py`

#### `run_pipeline(bids_root: str, config_path: str)`

Single entry point. Responsibilities in order:

1. Parse YAML (only here — no other module reads the file).
2. Instantiate `BIDSIndex`, `ReferenceResolver`, `TemplateLoader`, `RegistrationPlanner`.
3. Call `extract_reference_map(config)`, `get_qmri_maps_grouped(config)`, `map_to_sources(config)`.
4. For each `(sub, ses)`, call `build_session_plan(...)` then `run_session(...)`.
5. Catch `RuntimeError` per session and log — a failing session does not abort the pipeline.

#### `run_session(plan, reg_config_template, reg_config_qmri)`

1. Parses both registration config dicts into `RegistrationConfig`.
2. Runs all `RegistrationJob` in `plan.registration_jobs` in order.
3. Stores output prefixes by `source_key`.
4. For each `ApplyTransformJob`, chains transforms in ANTs order (most recent first):
   - `T(ref→template) 1Warp`
   - `T(ref→template) 0GenericAffine`
   - `T(source→ref) 0GenericAffine`

---

## Data flow

```
config.yaml
    │
    ▼
run_pipeline()
    ├── BIDSIndex.get_qmri_maps_grouped(config)  → grouped qmaps
    ├── BIDSIndex.map_to_sources(config)          → source map
    ├── ReferenceResolver.extract_reference_map(config) → reference paths
    └── for each (sub, ses):
            │
            ▼
        RegistrationPlanner.build_session_plan()
            │  produces SessionPlan
            ▼
        run_session(plan)
            ├── run_registration(ref → template)       [SyN]
            ├── run_registration(source_A → ref)       [Rigid+Affine]
            ├── run_registration(source_B → ref)       [Rigid+Affine]
            └── antsApplyTransforms per qmap
                    T(source→ref) ∘ T(ref→template)
```

---

## Configuration contract

The full config dict passed around the codebase has this structure:

```python
{
    "reference": {
        "suffix": str,       # required
        "acq": str,          # optional
        "run": str,          # optional
        "res": str,          # optional
        # any other BIDS entity
    },
    "template": {
        "atlas": str,        # required — must match a folder in resources/
    },
    "registration": {
        "ref_to_template": { "stages": [...] },
        "ref_to_qmri":     { "stages": [...] },
    },
    "filter": {              # optional block
        "subjects": list[str] | None,
        "sessions": list[str] | None,
    }
}
```

No module outside `runner.py` reads the raw YAML. Every module receives the parsed dict and accesses only its own key.

---

## Adding a new atlas

1. Create a folder under `registrabids/resources/<AtlasName>/`.
2. Add at minimum a file named `<AtlasName>_template.nii.gz`.
3. Optionally add `<AtlasName>_mask.nii.gz` and `<AtlasName>_annotation.nii.gz` (or `_dseg.nii.gz`).
4. Declare it in `pyproject.toml` package data if not already covered by the glob:
   ```toml
   [tool.setuptools.package-data]
   registrabids = ["resources/**/*.nii.gz"]
   ```
5. Reference it in config: `template: {atlas: AtlasName}`.

No code changes required.

---

## Adding a new registration stage type

ANTs transform types (e.g. `BSplineSyN`, `Similarity`, `Translation`) are passed through as strings in `StageConfig.transform` — no enum, no validation against a fixed list. To use a new transform:

1. Add it to your YAML `stages` block with the correct `transform_params`.
2. `StageConfig.transform_string` will produce `BSplineSyN[...]` automatically.

If the new transform requires additional flags not covered by the current command builder, extend `_build_command` in `registration.py` with a conditional on `stage.transform`.

---

## Testing

```bash
# Run all tests
pixi run pytest tests/ -v

# Run a specific file
pixi run pytest tests/test_planner.py -v

# With coverage
pixi run pytest tests/ --cov=registrabids --cov-report=term-missing
```

Tests use `unittest.mock.MagicMock` to simulate `BIDSImageFile` objects — no real BIDS dataset is required. The `tmp_path` pytest fixture provides temporary directories for output path assertions.

When adding a new feature, the minimum expected test coverage is:

- Happy path with representative inputs.
- Edge case: empty inputs (no qmaps, no sessions).
- Error path: missing required fields raise the correct exception type.

---

## Logging conventions

All modules use `logging.getLogger(__name__)`. Log levels follow this convention:

| Level | Usage |
|---|---|
| `DEBUG` | Detailed per-file information (paths, keys, discovered pairs) |
| `INFO` | Per-session progress, counts, major steps |
| `WARNING` | Recoverable issues (missing source, multiple candidates, missing mask) |
| `ERROR` | Non-fatal failures (session skipped, file not found) |

Never use `print()` in library code. The CLI configures the root logger level from `--log-level`.

---

## Dependency management with pixi

```bash
# Install all dependencies
pixi install

# Add a new dependency
pixi add <package>

# Add a dev-only dependency
pixi add --feature dev <package>

# Run a command in the pixi environment
pixi run <command>
```

---

## Contributing

1. Fork the repository and create a feature branch: `git checkout -b feat/my-feature`.
2. Follow the architecture principles described above — in particular, no module should parse the YAML config file.
3. Add tests for any new public method.
4. Run the test suite before opening a pull request: `pixi run pytest tests/ -v`.
5. Use descriptive commit messages: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.

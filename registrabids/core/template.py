from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the resources folder
_RESOURCES_DIR = Path(__file__).parent.parent / "resources"

# Useful suffixes in the ABAv3 atlas
_KNOWN_SUFFIXES = {
    "template": ["_anat.nii.gz"],
    "mask":     ["_mask.nii.gz"],
    "annotation": ["_dseg.nii.gz"],
}

@dataclass
class AtlasFiles:
    """Initializes the atlas by verifying that the necessary files exist."""
    name: str
    template: Path
    mask: Path | None = None
    annotation: Path | None = None
    extra: dict[str, Path] = field(default_factory=dict)

    def __post_init__(self): 
        if not self.template.exists():
            raise FileNotFoundError(
                f"Template file not found for the atlas '{self.name}' : "
                f"{self.template}\n"
                f"Check the contents of {self.template.parent}"
            )
        if self.mask and not self.mask.exists():
            logger.warning(
                "Mask listed but not found for '%s' : %s",
                self.name, self.mask
            )
            self.mask = None


class TemplateLoader:
    """
    Resolves and displays the files in an atlas stored in the `resources/` directory.

    Usage :
        loader = TemplateLoader()
        atlas = loader.load("ABAv3")
        print(atlas.template) 
    """

    def __init__(self, resources_dir: Path | None = None):
        self.resources_dir = resources_dir or _RESOURCES_DIR
        if not self.resources_dir.exists():
            raise FileNotFoundError(
                f"Directory “resources” not found: {self.resources_dir}"
            )

    # ─────────────────────────────────────────
    # public API 
    # ─────────────────────────────────────────

    def available_atlases(self) -> list[str]:
        """Lists the atlases available in resources/."""
        return [
            d.name for d in sorted(self.resources_dir.iterdir())
            if d.is_dir()
        ]

    def load(self, atlas_name: str) -> AtlasFiles:
        """
        Loads an atlas by name (= name of the subfolder in resources/).
        Throws a clear error if the atlas is unknown or incomplete.
        """
        atlas_dir = self.resources_dir / atlas_name

        if not atlas_dir.exists():
            available = self.available_atlases()
            raise ValueError(
                f"Atlas '{atlas_name}' not found in {self.resources_dir}.\n"
                f"Atlas availables : {available if available else '(none)'}"
            )

        template = self._find_file(atlas_dir, atlas_name, _KNOWN_SUFFIXES["template"])
        mask = self._find_file(
            atlas_dir, atlas_name, _KNOWN_SUFFIXES["mask"], required=False
        )
        annotation = self._find_file(
            atlas_dir, atlas_name, _KNOWN_SUFFIXES["annotation"], required=False
        )

        # Any other .nii.gz files in the folder → stored in “extra”
        known = {f for f in [template, mask, annotation] if f is not None}
        extra = {
            p.stem.split(".")[0]: p
            for p in atlas_dir.glob("*.nii.gz")
            if p not in known
        }

        atlas = AtlasFiles(
            name=atlas_name,
            template=template,
            mask=mask,
            annotation=annotation,
            extra=extra,
        )

        logger.info(
            "Atlas '%s' loaded : template=%s | mask=%s | annotation=%s | extra=%s",
            atlas_name,
            atlas.template.name,
            atlas.mask.name if atlas.mask else "—",
            atlas.annotation.name if atlas.annotation else "—",
            list(extra.keys()) or "—",
        )
        return atlas

    @classmethod
    def from_config(cls, config: dict) -> AtlasFiles:
        """
        Built directly from the YAML `template` block.

        config = {“atlas”: “ABAv3”}
        """
        atlas_name = config.get("atlas")
        if not atlas_name:
            raise ValueError(
                "The `template` block in the YAML file must contain an `atlas` key."
            )
        return cls().load(atlas_name)

    # ─────────────────────────────────────────
    # Helpers (private)
    # ─────────────────────────────────────────

    def _find_file(
        self,
        atlas_dir: Path,
        atlas_name: str,
        suffixes: list[str],
        required: bool = True,
    ) -> Path | None:
        """
        Find the first file in atlas_dir whose name
        matches atlas_name plus one of the known suffixes.
        """
        for suffix in suffixes:
            candidate = atlas_dir / f"{atlas_name}{suffix}"
            if candidate.exists():
                return candidate

        # Fallback: Search for any file with this extension in the folder
        for suffix in suffixes:
            matches = list(atlas_dir.glob(f"*{suffix}"))
            if matches:
                if len(matches) > 1:
                    logger.warning(
                        "Several files '%s' found for '%s', "
                        "use of : %s",
                        suffix, atlas_name, matches[0].name
                    )
                return matches[0]

        if required:
            raise FileNotFoundError(
                f"No template file found for the atlas '{atlas_name}' "
                f"in {atlas_dir}.\n"
                f"Expected suffixes : {suffixes}\n"
                f"Contents of the file : {[f.name for f in atlas_dir.iterdir()]}"
            )
        return None
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Lucas ARCAMONE

from bids import BIDSLayout
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BIDSIndex:
    def __init__(self, root):
        self.layout = BIDSLayout(
            root, derivatives=[f"{root}/derivatives/qmri"], validate=False
        )

    def get_subjects(self):
        return self.layout.get_subjects()

    def get_derivatives(self):
        return self.layout.get(scope="derivatives")

    # ─────────────────────────────────────────
    # MÉTHODES PRINCIPALES — basées sur config
    # ─────────────────────────────────────────

    def get_qmri_maps(self, config: dict, suffixes=None):
        """
        Récupère les qmri maps filtrées selon config['filter'] si présent.
        suffixes : optionnel pour filtrer par suffix (ex: ['MTRmap', 'MTsat'])
        """
        filter_cfg = config.get("filter", {})
        subjects = filter_cfg.get("subjects")
        sessions = filter_cfg.get("sessions")

        query = {"scope": "derivatives", "extension": [".nii", ".nii.gz"]}

        if subjects is not None:
            query["subject"] = subjects
        if sessions is not None:
            query["session"] = sessions
        if suffixes is not None:
            query["suffix"] = suffixes

        files = self.layout.get(**query)

        # Filtre les fichiers sans subject/session (dataset-level)
        valid_files = [
            f
            for f in files
            if f.entities.get("subject") is not None
            and f.entities.get("session") is not None
        ]

        if len(valid_files) < len(files):
            logger.debug(
                "%d fichier(s) qmri sans subject/session ignoré(s)",
                len(files) - len(valid_files),
            )

        return valid_files

    def get_qmri_maps_grouped(self, config: dict):
        """
        Retourne les qmri maps groupées par (subject, session).
        {('M30', '01'): [BIDSImageFile, ...], ...}
        """
        files = self.get_qmri_maps(config)
        grouped = defaultdict(list)

        for f in files:
            sub = f.entities.get("subject")
            ses = f.entities.get("session")
            # Déjà filtré dans get_qmri_maps, mais guard par sécurité
            if sub is not None and ses is not None:
                grouped[(sub, ses)].append(f)

        logger.info("%d couple(s) (sub, ses) avec qmri maps trouvé(s)", len(grouped))
        return grouped

    def map_to_sources(self, config: dict):
        """
        Retourne le mapping {qmap_path: [source_path, ...]}.
        """
        files = self.get_qmri_maps(config)

        mapping = {}
        root = Path(self.layout.root)

        for f in files:
            metadata = f.get_metadata()
            srcs = metadata.get("Sources", [])

            if isinstance(srcs, str):
                srcs = [srcs]
            resolved = []

            for src in srcs:
                src_path = Path(src)
                if not src_path.is_absolute():
                    # Resolves the relative path from the root of the dataset
                    src_path = (root / src_path).resolve()
                    logger.debug(
                        "Resolved relative path : %s → %s",
                        src,
                        src_path,
                    )
                if not src_path.exists():
                    logger.warning(
                        "Source not found for %s : %s",
                        Path(f.path).name,
                        src_path,
                    )
                resolved.append(str(src_path))
            mapping[f.path] = list(set(srcs))

        logger.debug("%d qmap(s) with extracted sources", len(mapping))
        return mapping

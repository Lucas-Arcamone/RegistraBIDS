import yaml
from pathlib import Path
from collections import defaultdict

class ReferenceResolver:

    def __init__(self, layout):
        self.layout = layout

    # -------------------------
    # LOAD CONFIG YAML
    # -------------------------
    def load_config(self, config_path):
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f)
        
    # -------------------------
    # MAIN FUNCTION
    # -------------------------
    def extract_reference_map(self, config_path, subjects=None, sessions=None):

        config = self.load_config(config_path)
        ref_config = config["reference"]

        reference_map = {}

        # boucle sujets/sessions présents dans le dataset
        files = self.layout.get(
            subject=subjects,
            session=sessions,
            extension=[".nii", ".nii.gz"]
        )

        grouped = defaultdict(list)

        for f in files:
            key = (f.entities.get("subject"), f.entities.get("session"))
            grouped[key].append(f)

        for (sub, ses), _ in grouped.items():

            query = {
                "subject": sub,
                "session": ses,
                "extension": [".nii", ".nii.gz"]
            }

            # inject YAML config (suffix, run, acq, res...)
            query.update({k: v for k, v in ref_config.items() if v is not None})

            candidates = self.layout.get(**query)

            if not candidates:
                raise ValueError(
                    f"No reference found for sub-{sub} ses-{ses} with {ref_config}"
                )

            reference_map[(sub, ses)] = [candidates[0].filename]

        return reference_map
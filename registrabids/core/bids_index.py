from bids import BIDSLayout
from collections import defaultdict

class BIDSIndex:
    def __init__(self, root):
        self.layout = BIDSLayout(root,
                                 derivatives=[f"{root}/derivatives/qmri"])

    def get_subjects(self):
        return self.layout.get_subjects()

    def get_derivatives(self):
        return self.layout.get(scope="derivatives")

    def get_qmri_maps(self, subjects=None, sessions=None, suffixes=None):
        query = {
            "scope": "qmri",
            "extension": [".nii", ".nii.gz"]
        }

        if subjects is not None:
            query["subject"] = subjects

        if sessions is not None:
            query["session"] = sessions

        if suffixes is not None:
            query["suffix"] = suffixes

        return self.layout.get(**query)    

    def get_qmri_maps_grouped(self, subjects=None, sessions=None):
        files = self.get_qmri_maps(subjects, sessions)

        grouped = defaultdict(list)

        for f in files:
            key = (f.entities.get("subject"), f.entities.get("session"))
            grouped[key].append(f)

        return grouped
    
    def map_to_sources(self, subjects=None, sessions=None):
        files = self.get_qmri_maps(subjects, sessions)

        mapping = {}

        for f in files:
            metadata = f.get_metadata()
            srcs = metadata.get("Sources", [])

            if isinstance(srcs, str):
                srcs = [srcs]

            mapping[f.path] = list(set(srcs))

        return mapping
    

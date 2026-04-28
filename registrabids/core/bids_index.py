from bids import BIDSLayout

class BIDSIndex:
    def __init__(self, root):
        self.layout = BIDSLayout(root, derivatives=True)

    def get_subjects(self):
        return self.layout.get_subjects()

    def get_derivatives(self):
        return self.layout.get(scope="derivatives")
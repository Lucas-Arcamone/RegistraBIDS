import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from registrabids.core.bids_index import BIDSIndex

bids_path = "/chemin/vers/ton/dataset"

index = BIDSIndex(bids_path)

print("Layout:", index.layout)
print("Subjects:", index.get_subjects())
print("Number of derivatives files:")
print(len(index.get_derivatives()))

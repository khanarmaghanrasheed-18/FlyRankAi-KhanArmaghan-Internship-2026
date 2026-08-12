"""Execute and save every revised notebook through ML-06."""

from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
NAMES = sys.argv[1:] or [
    "w01_research_question.ipynb",
    "w02_ml_task_framing.ipynb",
    "w03_data_contract.ipynb",
    "w03_feature_leakage_check.ipynb",
    "w04_signal_audit.ipynb",
]

for name in NAMES:
    path = ROOT / "work" / "notebooks" / name
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"PASS {name}")

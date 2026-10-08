"""The Colab notebook's install cell must be a valid requirement on current pip (>= 25)."""

import json
from pathlib import Path


def test_notebook_install_cell_uses_direct_url_requirement():
    nb = json.loads((Path(__file__).resolve().parent.parent / "notebooks" / "demo_run_pipeline.ipynb").read_text())
    installs = [
        "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "pip install" in "".join(c["source"])
    ]
    assert installs, "no install cell found"
    for src in installs:
        assert "#egg=" not in src, "pip >= 25 rejects extras in #egg fragments"
        assert "perturbseq-pipeline[demo] @ git+https://github.com/weili-lab/perturbseq-pipeline.git" in src

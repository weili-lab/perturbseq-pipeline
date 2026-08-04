"""Download the small-scale demo dataset and write a ready-to-run config.

The demo is one lane of a human ESC transcription-factor Perturb-seq screen
(416 guides against 61 targets, including 30 non-targeting controls), shared as
a public Google Drive folder.

    python demo/fetch_demo_data.py --dest demo_data --write-config config/demo.local.yaml
    perturbseq-pipeline run --config config/demo.local.yaml

The script locates the 10x directories inside whatever was downloaded and writes
their real paths into the config, so it keeps working if the folder layout
changes. Use ``--source`` to point at a copy you already have (e.g. a mounted
Drive path) and skip the download entirely.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Dict, List

DEMO_FOLDER_URL = "https://drive.google.com/drive/folders/1tU89UlsmZ6qTKPj348decXm523McLpvo"

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA = REPO_ROOT / "demo" / "sample_metadata.csv"


def find_mtx_dirs(root: Path) -> Dict[str, Path]:
    """Find every 10x MTX directory under ``root``, keyed by a lane ID."""
    found: Dict[str, Path] = {}
    for matrix in sorted(root.rglob("matrix.mtx*")):
        d = matrix.parent
        has_barcodes = any((d / f"barcodes.tsv{s}").is_file() for s in ("", ".gz"))
        has_features = any(
            (d / f"{n}.tsv{s}").is_file()
            for n in ("features", "genes")
            for s in ("", ".gz")
        )
        if not (has_barcodes and has_features):
            continue
        lane = d.name
        for prefix in ("filtered_feature_bc_matrix_", "raw_feature_bc_matrix_"):
            if lane.startswith(prefix):
                lane = lane[len(prefix) :]
        found[lane or d.name] = d
    return found


def _drive_folder_name() -> str | None:
    """Look up the name of the shared Drive folder without downloading it.

    ``gdown.download_folder(output=...)`` writes the folder *contents* straight
    into the output directory, discarding the folder name. That name is the lane
    ID (``filtered_feature_bc_matrix_S1lane1``), so it is fetched first and used
    to create a properly named subdirectory.
    """
    import os
    import tempfile

    import gdown

    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            listing = gdown.download_folder(
                DEMO_FOLDER_URL, skip_download=True, quiet=True, use_cookies=False
            )
        finally:
            os.chdir(cwd)
        if not listing:
            return None
        rel = Path(listing[0].local_path)
        if rel.is_absolute():
            try:
                rel = rel.relative_to(Path(tmp).resolve())
            except ValueError:
                rel = Path(rel.name)
        return rel.parts[0] if len(rel.parts) > 1 else None


def download(dest: Path) -> None:
    """Fetch the shared Drive folder with gdown, preserving its folder name."""
    try:
        import gdown
    except ImportError:
        sys.exit(
            "gdown is required to download the demo data.\n"
            "  pip install gdown\n"
            "Or pass --source PATH if you already have the data locally."
        )

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading demo data from {DEMO_FOLDER_URL}\n  into {dest} ...")
    try:
        name = _drive_folder_name()
        target = dest / name if name else dest
        if name:
            target.mkdir(parents=True, exist_ok=True)
        gdown.download_folder(
            DEMO_FOLDER_URL, output=str(target), quiet=False, use_cookies=False
        )
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit(
            f"\nDownload failed: {exc}\n\n"
            "The demo folder must be shared as 'Anyone with the link' for this to\n"
            "work. If you have the data already (for example on a mounted Drive),\n"
            "skip the download:\n"
            "  python demo/fetch_demo_data.py --source '/path/to/raw_counts'\n"
        )


def copy_from_source(source: Path, dest: Path) -> None:
    """Copy 10x directories from a local/mounted source instead of downloading."""
    dirs = find_mtx_dirs(source)
    if not dirs:
        sys.exit(f"No 10x MTX directories found under {source}")
    dest.mkdir(parents=True, exist_ok=True)
    for lane, d in dirs.items():
        target = dest / d.name
        if target.exists():
            print(f"  {lane}: already present at {target}")
            continue
        print(f"  {lane}: copying {d} -> {target}")
        shutil.copytree(d, target)


def write_config(mtx_dirs: Dict[str, Path], path: Path, metadata: Path, outdir: str) -> None:
    """Write a runnable config with the discovered paths filled in."""
    import yaml

    from perturbseq_pipeline.config import Config

    template = REPO_ROOT / "config" / "demo.yaml"
    cfg = Config.from_yaml(template) if template.is_file() else Config()
    cfg.input.mode = "mtx"
    cfg.input.mtx_dirs = {lane: str(p) for lane, p in sorted(mtx_dirs.items())}
    cfg.run.outdir = outdir
    cfg.metadata.file = str(metadata) if metadata.is_file() else None

    # Catch a lane/metadata mismatch now rather than part-way through a run.
    if cfg.metadata.file:
        import pandas as pd

        known = set(pd.read_csv(metadata)[cfg.metadata.key_column].astype(str))
        missing = sorted(set(mtx_dirs) - known)
        if missing and len(mtx_dirs) == 1:
            print(
                f"\nNote: lane {missing[0]!r} has no row in {metadata.name}; "
                "running without sample metadata (allowed for a single lane)."
            )
            cfg.metadata.file = None
        elif missing:
            sys.exit(
                f"\nLane(s) {missing} have no row in {metadata}.\n"
                f"Multi-lane runs require metadata for every lane. Add the rows, "
                f"or pass --metadata with a file that covers them."
            )

    cfg.validate()
    cfg.dump_yaml(path)
    print(f"\nWrote config: {path}")
    print(yaml.safe_dump({"input": {"mtx_dirs": cfg.input.mtx_dirs}}, sort_keys=False))


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="demo_data", help="where to put the data (default: demo_data)")
    p.add_argument("--source", default=None, help="copy from this local path instead of downloading")
    p.add_argument("--write-config", default="config/demo.local.yaml", help="config file to write")
    p.add_argument("--outdir", default="results/demo", help="run.outdir for the written config")
    p.add_argument("--metadata", default=str(DEFAULT_METADATA), help="sample metadata CSV")
    args = p.parse_args(argv)

    dest = Path(args.dest)
    if args.source:
        copy_from_source(Path(args.source), dest)
    elif dest.exists() and find_mtx_dirs(dest):
        print(f"Demo data already present in {dest}; skipping download.")
    else:
        download(dest)

    mtx_dirs = find_mtx_dirs(dest)
    if not mtx_dirs:
        sys.exit(
            f"No 10x MTX directories found under {dest} after fetching. "
            "Expected folder(s) containing barcodes.tsv.gz, features.tsv.gz and "
            "matrix.mtx.gz."
        )
    print(f"\nFound {len(mtx_dirs)} lane(s):")
    for lane, d in sorted(mtx_dirs.items()):
        size = sum(f.stat().st_size for f in d.iterdir() if f.is_file()) / 1e6
        print(f"  {lane:12s} {d}  ({size:.0f} MB)")

    write_config(mtx_dirs, Path(args.write_config), Path(args.metadata), args.outdir)
    print("Now run:\n  perturbseq-pipeline run --config " + args.write_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

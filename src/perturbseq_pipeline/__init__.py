"""A simple, reproducible Perturb-seq QC / clustering / perturbation pipeline.

Typical use from a notebook::

    from perturbseq_pipeline import Config, run_pipeline

    cfg = Config.from_yaml("config/demo.yaml")
    result = run_pipeline(cfg)
    print(result.report)
"""

__version__ = "0.1.0"

from .config import Config  # noqa: E402,F401

__all__ = ["Config", "run_pipeline", "PipelineResult", "__version__"]


def __getattr__(name: str):
    # Deferred so that ``import perturbseq_pipeline`` stays cheap and does not
    # pull in scanpy/matplotlib until a pipeline is actually run.
    if name in ("run_pipeline", "PipelineResult"):
        from . import cli

        return getattr(cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

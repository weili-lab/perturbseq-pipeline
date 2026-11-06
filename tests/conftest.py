"""Shared pytest fixtures."""

from __future__ import annotations

import logging

import pytest

PACKAGE_LOGGER = "perturbseq_pipeline"

# Pristine state, captured once before any test (or fixture) has run the pipeline.
_logger = logging.getLogger(PACKAGE_LOGGER)
_BASELINE = (list(_logger.handlers), _logger.level, _logger.propagate)


def _reset_package_logger() -> None:
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        if handler not in _BASELINE[0]:
            logger.removeHandler(handler)
            handler.close()
    logger.handlers[:] = _BASELINE[0]
    logger.setLevel(_BASELINE[1])
    logger.propagate = _BASELINE[2]


@pytest.fixture(autouse=True)
def _isolate_package_logger():
    """Give every test the pristine package logger, and restore it afterwards.

    ``run_pipeline`` calls ``cli.setup_logging``, which replaces the
    ``perturbseq_pipeline`` logger's handlers (console + a FileHandler in the
    run's tmp directory) and sets ``propagate = False`` -- correct for the CLI,
    but process-global. Left alone, every later test inherits that state:
    ``caplog`` (which listens on the root logger) stops seeing package
    warnings, so a ``caplog`` assertion passes alone and fails after any
    end-to-end test, and the FileHandlers of earlier runs stay open.

    The reset runs *before* the test as well as after, against a baseline
    captured at import: module-scoped fixtures that call ``run_pipeline`` are
    set up before this function-scoped fixture, so restoring a state saved at
    test start would just re-apply their leak.
    """
    _reset_package_logger()
    try:
        yield
    finally:
        _reset_package_logger()

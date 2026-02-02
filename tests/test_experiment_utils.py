from __future__ import annotations

from datetime import datetime, timezone
import io
import random

import numpy as np

from shap_stability.experiment_utils import (
    configure_logging,
    create_run_metadata,
    generate_run_id,
    set_global_seed,
)


def test_generate_run_id_includes_prefix_and_timestamp() -> None:
    fixed = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    run_id = generate_run_id(prefix="demo", now=fixed)
    assert run_id.startswith("demo-20250102-030405-")


def test_set_global_seed_reproducible() -> None:
    set_global_seed(123)
    first_random = random.random()
    first_numpy = np.random.rand()

    set_global_seed(123)
    second_random = random.random()
    second_numpy = np.random.rand()

    assert first_random == second_random
    assert first_numpy == second_numpy


def test_create_run_metadata_contains_core_fields() -> None:
    fixed = datetime(2025, 2, 3, 4, 5, 6, tzinfo=timezone.utc)
    metadata = create_run_metadata(
        run_id="run-1",
        seed=7,
        now=fixed,
        extra={"extra": "value"},
    )
    assert metadata["run_id"] == "run-1"
    assert metadata["seed"] == 7
    assert metadata["started_at"] == "2025-02-03T04:05:06+00:00"
    assert metadata["extra"] == "value"


def test_configure_logging_includes_run_context() -> None:
    stream = io.StringIO()
    logger = configure_logging(
        run_id="run-123",
        seed=42,
        stream=stream,
        force=True,
        logger_name="test-logger",
    )
    logger.info("hello")
    output = stream.getvalue()
    assert "run_id=run-123" in output
    assert "seed=42" in output
    assert "hello" in output


def test_configure_logging_updates_run_context() -> None:
    stream = io.StringIO()
    logger = configure_logging(
        run_id="run-a",
        seed=1,
        stream=stream,
        force=True,
        logger_name="test-logger-reconfigure",
    )
    logger.info("first")

    configure_logging(
        run_id="run-b",
        seed=2,
        stream=stream,
        logger_name="test-logger-reconfigure",
    )
    logger.info("second")

    output = stream.getvalue()
    assert "run_id=run-a" in output
    assert "seed=1" in output
    assert "run_id=run-b" in output
    assert "seed=2" in output

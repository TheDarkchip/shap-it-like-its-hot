"""Dataset acquisition and loading utilities."""

from .german_credit import (
    GERMAN_CREDIT_COLUMNS,
    GERMAN_CREDIT_URL,
    download_german_credit,
    load_german_credit,
)

__all__ = [
    "GERMAN_CREDIT_COLUMNS",
    "GERMAN_CREDIT_URL",
    "download_german_credit",
    "load_german_credit",
]

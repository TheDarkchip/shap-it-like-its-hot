"""Download the German Credit dataset to the local cache."""

from pathlib import Path

from shap_stability.data import download_german_credit  # noqa: E402


def main() -> None:
    path = download_german_credit()
    print(path)


if __name__ == "__main__":
    main()

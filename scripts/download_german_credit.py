"""Download the German Credit dataset to the local cache."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from german_credit import download_german_credit  # noqa: E402


def main() -> None:
    path = download_german_credit()
    print(path)


if __name__ == "__main__":
    main()

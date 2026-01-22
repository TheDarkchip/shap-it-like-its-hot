"""Download the German Credit dataset to the local cache."""

from shap_it_like_its_hot.data.german_credit import download_german_credit


def main() -> None:
    path = download_german_credit()
    print(path)


if __name__ == "__main__":
    main()

"""Download the MNIST pickle used by the original Nielsen examples."""

from pathlib import Path
from urllib.request import urlretrieve


DATASET_URL = (
    "https://raw.githubusercontent.com/mnielsen/"
    "neural-networks-and-deep-learning/master/data/mnist.pkl.gz"
)
DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "mnist.pkl.gz"


def main():
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATASET_PATH.exists():
        print(f"Dataset already exists at {DATASET_PATH}")
        return

    print(f"Downloading MNIST to {DATASET_PATH}")
    urlretrieve(DATASET_URL, DATASET_PATH)
    print("Download complete")


if __name__ == "__main__":
    main()

"""Fashion-MNIST loading with pinned source URLs and SHA-256 verification.

The files are fetched from the official zalandoresearch/fashion-mnist GitHub
repository at a fixed commit, so the exact bytes used in this reproduction are
pinned. Data is cached under ``data/`` (git-ignored).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

FMNIST_COMMIT = "b2617bb6d3ffa2e429640350f613e3291e10b141"
FMNIST_BASE_URL = (
    f"https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/{FMNIST_COMMIT}/data/fashion/"
)
FMNIST_FILES = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
# Mean / std of the Fashion-MNIST training images (pixel values scaled to [0, 1]).
MEAN, STD = 0.2860, 0.3530


def default_data_dir() -> Path:
    return Path(os.environ.get("CALREPRO_DATA", "data")) / "fashion-mnist"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_fashion_mnist(data_dir: Path | None = None) -> Path:
    """Download (if needed) and SHA-256-verify the four Fashion-MNIST files."""
    data_dir = Path(data_dir or default_data_dir())
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FMNIST_FILES.items():
        path = data_dir / name
        if not path.exists() or _sha256(path) != expected:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(FMNIST_BASE_URL + name, tmp)
            got = _sha256(tmp)
            if got != expected:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"SHA-256 mismatch for {name}: {got} != {expected}")
            tmp.replace(path)
    return data_dir


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


@dataclass
class Splits:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def load_splits(
    n_val: int = 5000,
    seed: int = 0,
    train_subset: int | None = None,
    test_subset: int | None = None,
    data_dir: Path | None = None,
) -> Splits:
    """Return normalized float tensors (N, 1, 28, 28) for train / val / test.

    The validation split (``n_val`` images, as in the paper's 5000-image
    CIFAR validation sets) is carved out of the 60k training set with a fixed
    permutation and is used only for post-hoc calibration.
    """
    d = download_fashion_mnist(data_dir)
    xtr = _read_idx(d / "train-images-idx3-ubyte.gz")
    ytr = _read_idx(d / "train-labels-idx1-ubyte.gz")
    xte = _read_idx(d / "t10k-images-idx3-ubyte.gz")
    yte = _read_idx(d / "t10k-labels-idx1-ubyte.gz")

    perm = np.random.RandomState(seed).permutation(len(xtr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if train_subset is not None:
        tr_idx = tr_idx[:train_subset]
    if test_subset is not None:
        xte, yte = xte[:test_subset], yte[:test_subset]

    def to_x(a: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(a.astype(np.float32) / 255.0).unsqueeze(1)
        return (t - MEAN) / STD

    def to_y(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a.astype(np.int64))

    return Splits(
        to_x(xtr[tr_idx]),
        to_y(ytr[tr_idx]),
        to_x(xtr[val_idx]),
        to_y(ytr[val_idx]),
        to_x(xte),
        to_y(yte),
    )

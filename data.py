"""Datasets as plain numpy arrays so they can be hashed identically everywhere."""
import numpy as np


def load_data(name="mnist", subset=None, seed=0, root="data"):
    """Returns (x_train, y_train, x_test, y_test), x as uint8 arrays."""
    if name == "mnist":
        from torchvision import datasets  # needs torchvision (see requirements.txt)

        tr = datasets.MNIST(root, train=True, download=True)
        te = datasets.MNIST(root, train=False, download=True)
        xtr, ytr = tr.data.numpy(), tr.targets.numpy()
        xte, yte = te.data.numpy(), te.targets.numpy()
    elif name == "synthetic":
        xtr, ytr = _synthetic(6000, seed)
        xte, yte = _synthetic(1000, seed + 1)
    else:
        raise ValueError(name)
    if subset:
        xtr, ytr = xtr[:subset], ytr[:subset]
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def _synthetic(n, seed):
    """10-class 8x8 'images': a fixed pattern per class + noise (for offline testing)."""
    protos = np.random.RandomState(1234).rand(10, 8, 8)
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 10, n)
    x = protos[y] + 0.35 * rng.randn(n, 8, 8)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8), y

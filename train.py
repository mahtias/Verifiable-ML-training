"""Train a classifier and record a signed, hash-chained provenance ledger.

    python train.py --run runs/run1                       # PyTorch + MNIST
    python train.py --backend numpy --dataset synthetic   # no torch needed (testing)
"""
import argparse
import platform
import sys
from pathlib import Path

import numpy as np

import provenance as pv
from data import load_data


# ---------------- backends ----------------
class NumpyBackend:
    """Softmax regression in plain numpy: lets the provenance pipeline be tested without torch."""
    ext = ".npz"

    def __init__(self, n_in, seed, lr):
        rng = np.random.RandomState(seed)
        self.W = 0.01 * rng.randn(n_in, 10)
        self.b = np.zeros(10)
        self.lr = lr

    def _p(self, x):
        z = x @ self.W + self.b
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def epoch(self, x, y, bs, rng):
        idx = rng.permutation(len(y))
        for i in range(0, len(y), bs):
            j = idx[i:i + bs]
            p = self._p(x[j])
            p[np.arange(len(j)), y[j]] -= 1
            self.W -= self.lr * x[j].T @ p / len(j)
            self.b -= self.lr * p.mean(0)

    def evaluate(self, x, y):
        p = self._p(x)
        loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
        return float(loss), float((p.argmax(1) == y).mean())

    def save(self, path):
        np.savez(path, W=self.W, b=self.b)


class TorchBackend:
    ext = ".pt"

    def __init__(self, n_in, seed, lr):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        self.torch = torch
        self.model = nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 10))
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.lossf = nn.CrossEntropyLoss()

    def epoch(self, x, y, bs, rng):
        t = self.torch
        xt, yt = t.from_numpy(x), t.from_numpy(y)
        idx = t.from_numpy(rng.permutation(len(y)))
        self.model.train()
        for i in range(0, len(y), bs):
            j = idx[i:i + bs]
            self.opt.zero_grad()
            self.lossf(self.model(xt[j]), yt[j]).backward()
            self.opt.step()

    def evaluate(self, x, y):
        t = self.torch
        self.model.eval()
        with t.no_grad():
            out = self.model(t.from_numpy(x))
            loss = self.lossf(out, t.from_numpy(y)).item()
            acc = (out.argmax(1) == t.from_numpy(y)).float().mean().item()
        return float(loss), float(acc)

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)


def prep(x):
    return (x.reshape(len(x), -1).astype(np.float32)) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/run1")
    ap.add_argument("--backend", choices=["torch", "numpy"], default="torch")
    ap.add_argument("--dataset", choices=["mnist", "synthetic"], default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=None, help="use first N training samples")
    a = ap.parse_args()
    dataset = a.dataset or ("mnist" if a.backend == "torch" else "synthetic")

    run = Path(a.run)
    if run.exists() and any(run.iterdir()):
        sys.exit(f"{run} already exists and is not empty; choose a new --run directory")
    run.mkdir(parents=True)
    sk = pv.generate_keys(run)
    led = pv.Ledger(run / "ledger.jsonl", sk)

    xtr, ytr, xte, yte = load_data(dataset, a.subset, a.seed)

    # 1. genesis: everything needed to reproduce the run
    led.append("genesis", {
        "backend": a.backend, "dataset": dataset, "subset": a.subset, "seed": a.seed,
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr,
        "train_py_sha256": pv.sha256_file(__file__),
        "python": platform.python_version(), "numpy": np.__version__,
    })
    # 2. dataset commitment (Merkle roots)
    led.append("dataset", {
        "train_merkle_root": pv.dataset_root(xtr, ytr), "n_train": int(len(ytr)),
        "test_merkle_root": pv.dataset_root(xte, yte), "n_test": int(len(yte)),
    })

    Xtr, Xte = prep(xtr), prep(xte)
    if a.backend == "torch":
        be = TorchBackend(Xtr.shape[1], a.seed, a.lr)
    else:
        be = NumpyBackend(Xtr.shape[1], a.seed, a.lr)
    rng = np.random.RandomState(a.seed)

    # 3. one signed entry per epoch, binding metrics to the exact checkpoint bytes
    for ep in range(1, a.epochs + 1):
        be.epoch(Xtr, ytr, a.batch_size, rng)
        loss, acc = be.evaluate(Xte, yte)
        ck = run / f"checkpoint_epoch{ep}{be.ext}"
        be.save(ck)
        led.append("epoch", {
            "epoch": ep, "test_loss": loss, "test_acc": acc,
            "checkpoint": ck.name, "checkpoint_sha256": pv.sha256_file(ck),
        })
        print(f"epoch {ep}: test_loss={loss:.4f} test_acc={acc:.4f}")

    # 4. final entry: the released model
    led.append("final", {
        "model": ck.name, "model_sha256": pv.sha256_file(ck), "test_acc": acc,
        "epochs_completed": a.epochs,
    })
    print(f"\nLedger head hash (publish/anchor this somewhere you trust):\n{led.head}")
    (run / "HEAD.txt").write_text(led.head + "\n")


if __name__ == "__main__":
    main()

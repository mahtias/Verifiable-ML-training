"""Train one run, then attack copies of it and check that every attack is detected.

    python tamper_test.py                 # uses torch+MNIST subset if torch is installed
    python tamper_test.py --backend numpy # offline, no torch
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import provenance as pv
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from verify import verify_run


def rewrite(run, entries):
    (run / "ledger.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))


def rehash(entries, sk=None):
    """Attacker recomputes the chain. Without the real key signatures can't be valid."""
    prev = pv.GENESIS_PREV
    for i, e in enumerate(entries):
        e["index"], e["prev_hash"] = i, prev
        body = {k: e[k] for k in ("index", "prev_hash", "kind", "payload", "timestamp")}
        e["hash"] = pv._entry_hash(body)
        if sk:
            e["signature"] = sk.sign(bytes.fromhex(e["hash"])).hex()
        prev = e["hash"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["torch", "numpy"], default=None)
    a = ap.parse_args()
    backend = a.backend
    if backend is None:
        try:
            import torch  # noqa: F401
            backend = "torch"
        except ImportError:
            backend = "numpy"
    tmp = Path(tempfile.mkdtemp())
    base = tmp / "base"
    cmd = [sys.executable, "train.py", "--run", str(base), "--backend", backend, "--epochs", "3"]
    cmd += ["--subset", "5000"] if backend == "torch" else []
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    anchor = (base / "HEAD.txt").read_text().strip()

    results = []

    def attack(name, fn, **kw):
        d = tmp / name.replace(" ", "_")
        shutil.copytree(base, d)
        extra = fn(d) or {}
        probs = verify_run(d, **{**kw, **extra})
        results.append((name, bool(probs), probs[:1]))

    def flip_ckpt(d):
        f = sorted(d.glob("checkpoint_epoch2*"))[0]
        b = bytearray(f.read_bytes()); b[len(b) // 2] ^= 0xFF; f.write_bytes(bytes(b))

    def edit_acc(d):
        e = pv.read_ledger(d / "ledger.jsonl"); e[-1]["payload"]["test_acc"] = 0.999; rewrite(d, e)

    def edit_rehash_nokey(d):
        e = pv.read_ledger(d / "ledger.jsonl"); e[-1]["payload"]["test_acc"] = 0.999
        rehash(e); rewrite(d, e)

    def delete_mid(d):
        e = pv.read_ledger(d / "ledger.jsonl"); del e[3]; rewrite(d, e)

    def truncate(d):
        e = pv.read_ledger(d / "ledger.jsonl"); rewrite(d, e[:-2])

    def resign_all(d):
        """Attacker rewrites history with their OWN key and swaps the public key."""
        sk = Ed25519PrivateKey.generate()
        e = pv.read_ledger(d / "ledger.jsonl"); e[-1]["payload"]["test_acc"] = 0.999
        rehash(e, sk); rewrite(d, e)
        from cryptography.hazmat.primitives import serialization as s
        (d / "public_key.pem").write_bytes(sk.public_key().public_bytes(
            s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))

    def poison(d):
        import numpy as np
        from data import load_data
        g = pv.read_ledger(d / "ledger.jsonl")[0]["payload"]
        x, y, _, _ = load_data(g["dataset"], g["subset"], g["seed"])
        y = y.copy(); y[:50] = (y[:50] + 1) % 10      # flip 50 labels
        return {"data": (x, y)}

    attack("clean run (should PASS)", lambda d: None, check_data=True)
    attack("flip one byte of a checkpoint", flip_ckpt)
    attack("edit accuracy in ledger", edit_acc)
    attack("edit + recompute hashes (no key)", edit_rehash_nokey)
    attack("delete a ledger entry", delete_mid)
    attack("truncate ledger tail", truncate)
    attack("truncate, checked only with anchor", truncate, anchor=anchor)
    attack("rewrite history with attacker key", resign_all, anchor=anchor)
    attack("poison 50 training labels", poison, check_data=True)

    ok = True
    print(f"\nbackend={backend}\n{'attack':42s} detected?")
    for name, detected, why in results:
        expected = not name.startswith("clean")
        good = detected == expected
        ok &= good
        print(f"{'OK ' if good else 'BAD'} {name:40s} {detected}  {why[0] if why else ''}")
    shutil.rmtree(tmp)
    print("\nALL ATTACKS BEHAVED AS EXPECTED" if ok else "\nSOMETHING WRONG")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

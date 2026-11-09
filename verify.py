"""Verify a training run: ledger chain + signatures + checkpoint files (+ dataset, + anchor).

    python verify.py --run runs/run1
    python verify.py --run runs/run1 --anchor <head hash you saved earlier> --check-data
"""
import argparse
import sys
from pathlib import Path

import provenance as pv


def verify_run(run, anchor=None, pubkey=None, check_data=False, data=None):
    """Return list of problems. `data`=(xtr,ytr) overrides loading (used by tests)."""
    run = Path(run)
    problems = []
    try:
        entries = pv.read_ledger(run / "ledger.jsonl")
    except Exception as ex:  # corrupted JSON counts as tampering
        return [f"ledger unreadable: {ex}"]
    pub = pv.load_public(pubkey or run / "public_key.pem")

    problems += pv.verify_ledger(entries, pub)

    kinds = [e.get("kind") for e in entries]
    if not entries or kinds[0] != "genesis":
        problems.append("first entry is not 'genesis'")
    if not entries or kinds[-1] != "final":
        problems.append("no 'final' entry: ledger is incomplete or was truncated")

    if anchor and (not entries or entries[-1].get("hash") != anchor):
        problems.append("ledger head does not match the anchored hash (rewritten or truncated)")

    for e in entries:
        p = e.get("payload", {})
        if e.get("kind") == "epoch":
            f = run / p["checkpoint"]
            if not f.exists():
                problems.append(f"checkpoint missing: {p['checkpoint']}")
            elif pv.sha256_file(f) != p["checkpoint_sha256"]:
                problems.append(f"checkpoint modified: {p['checkpoint']}")
        if e.get("kind") == "final":
            f = run / p["model"]
            if not f.exists() or pv.sha256_file(f) != p["model_sha256"]:
                problems.append(f"final model does not match ledger: {p['model']}")

    if check_data and entries and len(entries) > 1:
        g, d = entries[0]["payload"], entries[1]["payload"]
        if data is None:
            from data import load_data
            xtr, ytr, _, _ = load_data(g["dataset"], g["subset"], g["seed"])
        else:
            xtr, ytr = data
        if pv.dataset_root(xtr, ytr) != d["train_merkle_root"]:
            problems.append("training data does not match the committed Merkle root (data changed/poisoned)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--anchor", help="expected head hash (from HEAD.txt you saved elsewhere)")
    ap.add_argument("--pubkey", help="trusted public key (default: the one in the run dir)")
    ap.add_argument("--check-data", action="store_true")
    a = ap.parse_args()
    probs = verify_run(a.run, a.anchor, a.pubkey, a.check_data)
    if probs:
        print("VERIFICATION FAILED")
        for p in probs:
            print("  -", p)
        sys.exit(1)
    print("VERIFIED: ledger chain, signatures and checkpoints are consistent.")


if __name__ == "__main__":
    main()

"""Tamper-evident training ledger: SHA-256 hash chain + Ed25519 signatures + Merkle roots.

Same primitives as a blockchain, no network needed:
  - each entry commits to the previous entry's hash (hash chain)
  - each entry hash is signed with the run's Ed25519 key
  - datasets are committed to by a Merkle root
"""
import hashlib
import json
import time
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature

GENESIS_PREV = "0" * 64


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- Merkle tree ----------
def merkle_root(leaves):
    """leaves: list of 32-byte digests. Odd node is promoted by duplication."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = [bytes(l) for l in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            hashlib.sha256(b"\x01" + level[i] + level[i + 1]).digest()
            for i in range(0, len(level), 2)
        ]
    return level[0].hex()


def dataset_leaves(x, y):
    """One leaf per sample: H(0x00 || sample bytes || label)."""
    return [
        hashlib.sha256(b"\x00" + x[i].tobytes() + int(y[i]).to_bytes(2, "little")).digest()
        for i in range(len(y))
    ]


def dataset_root(x, y):
    return merkle_root(dataset_leaves(x, y))


# ---------- keys ----------
def generate_keys(run_dir):
    run_dir = Path(run_dir)
    sk = Ed25519PrivateKey.generate()
    (run_dir / "signing_key.pem").write_bytes(
        sk.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    (run_dir / "public_key.pem").write_bytes(
        sk.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    return sk


def load_private(path):
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def load_public(path) -> Ed25519PublicKey:
    return serialization.load_pem_public_key(Path(path).read_bytes())


# ---------- ledger ----------
def _entry_hash(body: dict) -> str:
    canon = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(canon)


class Ledger:
    def __init__(self, path, signing_key):
        self.path = Path(path)
        self.sk = signing_key
        self.entries = []

    def append(self, kind: str, payload: dict) -> dict:
        body = {
            "index": len(self.entries),
            "prev_hash": self.entries[-1]["hash"] if self.entries else GENESIS_PREV,
            "kind": kind,
            "payload": payload,
            "timestamp": time.time(),
        }
        h = _entry_hash(body)
        entry = dict(body, hash=h, signature=self.sk.sign(bytes.fromhex(h)).hex())
        self.entries.append(entry)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    @property
    def head(self):
        return self.entries[-1]["hash"] if self.entries else GENESIS_PREV


def read_ledger(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def verify_ledger(entries, public_key):
    """Return list of problems (empty list = chain and signatures are valid)."""
    problems = []
    prev = GENESIS_PREV
    if not entries:
        return ["ledger is empty"]
    for i, e in enumerate(entries):
        if e.get("index") != i:
            problems.append(f"entry {i}: index is {e.get('index')} (entry deleted or reordered)")
        if e.get("prev_hash") != prev:
            problems.append(f"entry {i}: prev_hash does not match previous entry (chain broken)")
        body = {k: e[k] for k in ("index", "prev_hash", "kind", "payload", "timestamp") if k in e}
        if _entry_hash(body) != e.get("hash"):
            problems.append(f"entry {i}: content does not match its hash (entry was edited)")
        try:
            public_key.verify(bytes.fromhex(e.get("signature", "")), bytes.fromhex(e.get("hash", "")))
        except (InvalidSignature, ValueError):
            problems.append(f"entry {i}: invalid signature (not signed by the trusted key)")
        prev = e.get("hash")
    return problems

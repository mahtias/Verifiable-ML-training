# Verifiable ML Training — tamper-evident provenance for model training

A small project that trains a classifier (PyTorch, MNIST) and records a **signed, hash-chained
ledger** of the whole run, so anyone can later check that a released model really comes from
the claimed data, code and training process — and that nothing was edited afterwards.

It reuses blockchain primitives (SHA-256 hash chain, Merkle trees, Ed25519 signatures) on an ML
pipeline. No network or chain is needed; the final ledger head hash can be anchored anywhere
(a git tag, an email, a smart contract).

## What is recorded
| Entry | Content |
|---|---|
| genesis | hyper-parameters, seed, SHA-256 of `train.py`, library versions |
| dataset | Merkle root of train and test sets (one leaf per sample + label) |
| epoch ×N | test loss/accuracy + SHA-256 of that epoch's checkpoint file |
| final | SHA-256 of the released model |

Each entry contains the previous entry's hash and is signed with Ed25519.

## Run (Linux / WSL)
```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python train.py --run runs/run1                  # trains, writes ledger, prints head hash
python verify.py --run runs/run1 --check-data    # -> VERIFIED
python tamper_test.py                            # runs 8 attacks, all must be detected
```
Without torch (testing only): `python train.py --backend numpy --run runs/np1`.

## Attacks tested (`tamper_test.py`)
flip a checkpoint byte · edit a value in the ledger · edit and recompute hashes without the key ·
delete an entry · truncate the tail · rewrite history with an attacker's own key ·
poison 50 training labels.

## Limits (be honest in an application)
- Proves **integrity and provenance**, not that the training was *correct* or the model is safe.
- The signing key sits in the run folder for the demo; in practice keep it in an HSM/KMS.
- Rewriting history with a new key is only caught if you **anchor the head hash** (`--anchor`) or
  pin the public key (`--pubkey`) somewhere the attacker cannot change.
- It trusts the machine that trains; it does not use zk-proofs or TEEs (natural future work:
  proof-carrying training, zkML).

## Ideas to extend
Differential-privacy noise logged in the ledger · federated rounds as ledger entries ·
anchor the head hash in an Ethereum contract · a model card generated from the ledger.
# Verifiable-ML-training

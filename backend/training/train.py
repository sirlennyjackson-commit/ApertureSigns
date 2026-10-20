"""Train a sign classifier from recorded landmark clips.

    python -m backend.training.train --data data/sequences --out models/signs.npz

Dataset layout (created by ``backend.training.collect``):

    data/sequences/<label>/<signer>__<timestamp>.npz

Accuracy is cross-validated *by signer* when 2+ signers exist (the honest
number: how well it works for someone it has never seen), otherwise stratified.
Augmentation is applied to training folds only, never to the held-out fold.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from backend.config.settings import get_settings
from backend.ml.augment import augment_sequence
from backend.ml.classifier import SignClassifier
from backend.ml.features import sequence_descriptor
from backend.ml.landmarks import load_sequence


def load_dataset(root: Path):
    seqs, labels, signers = [], [], []
    for path in sorted(root.glob("*/*.npz")):
        try:
            seq = load_sequence(path)
        except (ValueError, OSError) as exc:
            print(f"  skipping {path}: {exc}", file=sys.stderr)
            continue
        seqs.append(seq)
        labels.append(path.parent.name)
        signers.append(path.stem.split("__", 1)[0] if "__" in path.stem else "unknown")
    return seqs, labels, signers


def build_xy(seqs, labels, *, augment: int, mirror_p: float, rng):
    X, y = [], []
    for seq, label in zip(seqs, labels, strict=True):
        X.append(sequence_descriptor(seq))
        y.append(label)
        for _ in range(augment):
            X.append(sequence_descriptor(augment_sequence(seq, rng, mirror_p=mirror_p)))
            y.append(label)
    return np.stack(X), np.array(y)


def fit(X: np.ndarray, y: np.ndarray, C: float = 0.1) -> SignClassifier:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, max_iter=3000).fit(scaler.transform(X), y)
    return SignClassifier(model.classes_, scaler.mean_, scaler.scale_, model.coef_, model.intercept_)


def cross_validate(seqs, labels, signers, *, folds, augment, mirror_p, C, rng):
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    idx = np.arange(len(seqs))
    unique_signers = sorted(set(signers))
    if len(unique_signers) >= 2:
        splitter = GroupKFold(n_splits=min(folds, len(unique_signers)))
        splits = splitter.split(idx, labels, groups=signers)
        mode = f"leave-signer-out ({min(folds, len(unique_signers))} folds)"
    else:
        smallest = min(Counter(labels).values())
        if smallest < 2:
            return None, "skipped (a class has fewer than 2 clips)", None
        n = min(folds, smallest)
        splits = StratifiedKFold(n_splits=n, shuffle=True, random_state=0).split(idx, labels)
        mode = f"stratified ({n} folds, single signer - optimistic!)"

    truth, guess = [], []
    for train_idx, test_idx in splits:
        X, y = build_xy([seqs[i] for i in train_idx], [labels[i] for i in train_idx],
                        augment=augment, mirror_p=mirror_p, rng=rng)
        model = fit(X, y, C)
        for i in test_idx:
            guess.append(model.predict(seqs[i]).label)
            truth.append(labels[i])
    accuracy = float(np.mean(np.array(truth) == np.array(guess)))
    recall = {
        lab: float(np.mean([g == lab for t, g in zip(truth, guess, strict=True) if t == lab]))
        for lab in sorted(set(truth))
    }
    return accuracy, mode, recall


def main(argv=None) -> int:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=settings.data_dir / "sequences")
    p.add_argument("--out", type=Path, default=settings.model_path)
    p.add_argument("--augment", type=int, default=8, help="augmented copies per clip")
    p.add_argument("--mirror", action="store_true", help="also mirror clips (left-handed signers)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--C", type=float, default=0.1, help="inverse regularisation strength")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-cv", action="store_true")
    args = p.parse_args(argv)

    seqs, labels, signers = load_dataset(args.data)
    counts = Counter(labels)
    if len(counts) < 2:
        print(f"Need at least 2 labels under {args.data}; found {dict(counts)}", file=sys.stderr)
        return 1
    print(f"{len(seqs)} clips, {len(counts)} labels, signers: {sorted(set(signers))}")
    for label, n in sorted(counts.items()):
        flag = "  <-- record more (aim for 20+)" if n < 10 else ""
        print(f"  {label:<20} {n:>4}{flag}")

    rng = np.random.default_rng(args.seed)
    mirror_p = 0.5 if args.mirror else 0.0
    cv_accuracy = None
    if not args.no_cv:
        cv_accuracy, mode, recall = cross_validate(
            seqs, labels, signers, folds=args.folds, augment=args.augment,
            mirror_p=mirror_p, C=args.C, rng=rng)
        if cv_accuracy is None:
            print(f"Cross-validation {mode}")
        else:
            print(f"\nCross-validation {mode}: accuracy {cv_accuracy:.1%}")
            for lab, r in recall.items():
                print(f"  {lab:<20} recall {r:.0%}")

    X, y = build_xy(seqs, labels, augment=args.augment, mirror_p=mirror_p, rng=rng)
    model = fit(X, y, args.C)
    model.meta.update(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        clips=dict(counts), signers=sorted(set(signers)), augment=args.augment,
        cv_accuracy=cv_accuracy, C=args.C,
    )
    out = model.save(args.out)
    print(f"\nSaved {out}  ({len(model.labels)} labels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Aperture Signs - v2 trainer.

    python backend\training\train_v2.py
    python backend\training\train_v2.py --min-examples 8
"""
import argparse, os, sys, warnings
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features_v2 import extract_many, FEATURE_VERSION

warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join("data", "processed", "sign_dataset.npz"))
ap.add_argument("--min-examples", type=int, default=5)
ap.add_argument("--out", default=os.path.join("models", "sign", "classifier"))
args = ap.parse_args()

d = np.load(args.data, allow_pickle=True)
X, y_old, groups, labels = d["X"], d["y"], d["groups"], [str(l) for l in d["labels"]]

# ---------- 1. clean ----------
names = np.array([labels[i] for i in y_old])
keep = np.array([not n.startswith(".") for n in names])
counts = {n: int((names[keep] == n).sum()) for n in set(names[keep])}
keep &= np.array([counts.get(n, 0) >= args.min_examples for n in names])
X, names, groups = X[keep], names[keep], groups[keep]
classes = sorted(set(names))
cid = {n: i for i, n in enumerate(classes)}
y = np.array([cid[n] for n in names])
K = len(classes)
print(f"Sequences: {len(y)} | Classes: {K} | Signers: {len(set(groups))}")

# ---------- 2. features ----------
F = extract_many(X)
print("Feature matrix:", F.shape)

# ---------- 3. models ----------
MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)),
    "svm_rbf": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True)),
    "extratrees": lambda: ExtraTreesClassifier(n_estimators=400, random_state=0, n_jobs=-1),
}
n_splits = max(2, min(5, int(np.bincount(y).min())))


def cv_proba(make, splitter):
    P = np.zeros((len(y), K))
    for tr, te in splitter.split(F, y, groups):
        m = make()
        m.fit(F[tr], y[tr])
        P[np.ix_(te, m.classes_ if hasattr(m, "classes_") else m[-1].classes_)] = m.predict_proba(F[te])
    return P


def topk(P, k):
    return float((np.argsort(-P, axis=1)[:, :k] == y[:, None]).any(axis=1).mean())


rand_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
sign_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)

lines, results = [], {}
lines.append(f"classes={K} sequences={len(y)} folds={n_splits}")
lines.append(f"chance top-1 = {1 / K:.3f}")
lines.append(f"{'model':12s} {'rand top1':>10s} {'rand top3':>10s} {'signer top1':>12s} {'signer top3':>12s}")
for name, make in MODELS.items():
    Pr, Ps = cv_proba(make, rand_cv), cv_proba(make, sign_cv)
    results[name] = (Pr, Ps)
    lines.append(f"{name:12s} {topk(Pr,1):10.3f} {topk(Pr,3):10.3f} {topk(Ps,1):12.3f} {topk(Ps,3):12.3f}")

best = max(results, key=lambda n: topk(results[n][1], 1))
Pr, Ps = results[best]
lines.append(f"\nBEST (by signer-independent top-1): {best}")

# ---------- 4. evaluation of best model (signer-independent) ----------
pred = Ps.argmax(1)
lines.append("\nPER-SIGN (signer-independent)\n" +
             classification_report(y, pred, labels=list(range(K)), target_names=classes, zero_division=0))

lines.append("REJECTION CURVE (signer-independent): confidence threshold -> coverage / accuracy on accepted")
conf = Ps.max(1)
threshold = 0.5
found = False
for t in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    m = conf >= t
    cov = m.mean()
    acc = (pred[m] == y[m]).mean() if m.any() else 0.0
    lines.append(f"  conf>={t:.2f}  coverage={cov:5.1%}  accuracy={acc:5.1%}")
    if not found and m.sum() >= 20 and acc >= 0.6:
        threshold, found = t, True
lines.append(f"chosen reject threshold: {threshold}")

cm = confusion_matrix(y, pred, labels=list(range(K)))
off = [(cm[i, j], classes[i], classes[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
lines.append("\nMOST CONFUSED PAIRS (true -> predicted)")
for n, a, b in sorted(off, reverse=True)[:15]:
    lines.append(f"  {a} -> {b}: {n}")

# ---------- 5. save ----------
os.makedirs(args.out, exist_ok=True)
final = MODELS[best]()
final.fit(F, y)
bundle = dict(model=final, classes=classes, feature_version=FEATURE_VERSION,
              sequence_length=30, threshold=threshold, model_name=best)
joblib.dump(bundle, os.path.join(args.out, "aperture_sign_v2.joblib"))
np.savetxt(os.path.join(args.out, "confusion_v2.csv"), cm, fmt="%d", delimiter=",",
           header=",".join(classes), comments="")
report = "\n".join(lines)
open(os.path.join(args.out, "report_v2.txt"), "w", encoding="utf-8").write(report)
print(report)
print("\nSAVED:", os.path.join(args.out, "aperture_sign_v2.joblib"))
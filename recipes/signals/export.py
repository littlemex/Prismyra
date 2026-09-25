"""Usage: export.py <problems.json> <labels.jsonl> <features.npz> <layer> <C> <probe.json> [model depth]

Fit the shipped self_solve probe on every labelled problem and fold it into one linear read: w . h + b on the last
context token of one layer. StandardScaler, PCA and the logistic layer are all affine, so their composition is a
single hidden-size vector and a bias -- the whole signal costs one dot product."""

import json, sys, hashlib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

problems, labels, feats, layer, C, out = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
    float(sys.argv[5]),
    sys.argv[6],
)
P = json.load(open(problems))
lab = {json.loads(l)["id"]: json.loads(l) for l in open(labels)}
z = np.load(feats)
layers = z["layers"].tolist()
ids = [str(i) for i in z["ids"]] if "ids" in z.files else [str(p["id"]) for p in P]
rows = [k for k, i in enumerate(ids) if int(i) in lab]  # problems whose labelling request errored are left out
# Unparsed answers (truncated at the token budget) count as failures: the target is "solved within this budget".
y = np.array([bool(lab[int(ids[k])]["greedy"]) for k in rows]).astype(int)
X = z["last"][rows, layers.index(layer)].astype(np.float64)
sc, pca = StandardScaler().fit(X), None
Xs = sc.transform(X)
pca = PCA(256, random_state=0).fit(Xs)
lr = LogisticRegression(C=C, max_iter=5000).fit(pca.transform(Xs), y)
w_pca = lr.coef_[0]  # 256
w_std = pca.components_.T @ w_pca  # hidden size, in standardised space
b_std = lr.intercept_[0] - pca.mean_ @ w_std  # PCA centres on its own mean
w = w_std / sc.scale_
b = b_std - (sc.mean_ / sc.scale_) @ w_std
check = X @ w + b
ref = lr.decision_function(pca.transform(Xs))
print("fold error", float(np.abs(check - ref).max()))
from sklearn.model_selection import KFold

cv = cross_val_score(
    make_pipeline(StandardScaler(), PCA(256, random_state=0), LogisticRegression(C=C, max_iter=5000)),
    X,
    y,
    cv=KFold(5, shuffle=True, random_state=0),
    scoring="roc_auc",
)
json.dump(
    {
        "name": "self_solve",
        "layer": layer,
        "position": "last_context_token",
        "weight": w.tolist(),
        "bias": float(b),
        "target": "greedy correctness of this base model answering the context as a chat prompt (thinking off, 1,536-token budget)",
        "trained_on": {
            "problems": len(y),
            "families": sorted(set(p["family"] for p in P)),
            "positive_rate": float(y.mean()),
        },
        "cv_auroc_overall": float(cv.mean()),
        "model": {
            "hidden_size": int(X.shape[1]),
            "layers": int(z["depth"]) if "depth" in z.files else int(sys.argv[7]),
        },
    },
    open(out, "w"),
)
print(json.dumps({"layer": layer, "cv_auroc": float(cv.mean()), "out": out}))

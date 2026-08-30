"""
Classification pipeline — Berlinet, Biau & Rouvière, Section 2.

Steps
-----
1. Optionally map each raw signal to its periodogram  (feature engineering).
2. Compute DWT coefficient matrix for all signals.
3. Rank basis functions by empirical energy on the training set  (eq. 2.4).
4. For each dimension d and each classifier, train on the first d
   reordered training coefficients.
5. Select the best (d, classifier) by minimising the empirical error
   on the validation set  (eq. 2.5).
6. Report classification metrics on the held-out test set.

The input representation is chosen in configs/config.yaml via ``input_domain``:
``raw`` (time domain), ``spectrum`` (periodogram) or ``both`` to compare them.
"""

import warnings
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from functional_supervised_classification.coeffient_compute import coeff_matrix, to_periodogram
from functional_supervised_classification.config import load_config
from functional_supervised_classification.data_loading import load_ecg200, load_phoneme
from functional_supervised_classification.data_visualisation import plot_wavelet_cluster_view

warnings.filterwarnings("ignore")

CFG     = load_config()
DATASET = CFG["dataset"]
DOMAIN  = CFG.get("input_domain", "raw")
assert DATASET in ["phoneme", "ecg200"], f"Dataset {DATASET} not supported. Choose 'phoneme' or 'ecg200'."
assert DOMAIN in ["raw", "spectrum", "both"], f"input_domain {DOMAIN} not supported. Choose 'raw', 'spectrum' or 'both'."

# ── Parameters ─────────────────────────────────────────────────────────────────

D_MAX    = 50   # upper bound on dimension search (W-QDA fails for large d)
VAL_FRAC = 0.3  # fraction of training data reserved for validation

CLASSIFIERS = {
    "W-NN"  : KNeighborsClassifier(),
    "W-QDA" : QuadraticDiscriminantAnalysis(),
    "W-CART": DecisionTreeClassifier(),
    "W-FFNN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

# feature engineering applied to the raw signals before the wavelet decomposition
TRANSFORMS = {"raw": lambda s: s, "spectrum": to_periodogram}

# ── Data loading and train/validation split ────────────────────────────────────

X_train_raw, y_train, X_test_raw, y_test = (load_phoneme if DATASET == "phoneme" else load_ecg200)()

S_train = X_train_raw[:, 0, :]
S_test  = X_test_raw[:, 0, :]

S_tr, S_val, y_tr, y_val = train_test_split(
    S_train, y_train, test_size=VAL_FRAC, random_state=42, stratify=y_train
)

if len(np.unique(y_test)) == 2:
    METRIC_KW = dict(average="binary", pos_label=sorted(np.unique(y_test))[-1], zero_division=0)
else:
    METRIC_KW = dict(average="macro", zero_division=0)


def run(domain: str) -> None:
    """Run the full pipeline (eq. 2.3–2.5) on the chosen input representation."""
    tf = TRANSFORMS[domain]
    C_tr, C_val, C_te = coeff_matrix(tf(S_tr)), coeff_matrix(tf(S_val)), coeff_matrix(tf(S_test))

    # energy-based ranking  (eq. 2.4)
    ranking = np.argsort(np.sum(C_tr ** 2, axis=0))[::-1]
    C_tr, C_val, C_te = C_tr[:, ranking], C_val[:, ranking], C_te[:, ranking]

    # joint selection of d and classifier  (eq. 2.5)
    best_err, best_d, best_name, best_clf = np.inf, None, None, None
    for d in range(1, min(D_MAX, C_tr.shape[1]) + 1):
        for name, clf in CLASSIFIERS.items():
            try:
                fitted = clone(clf).fit(C_tr[:, :d], y_tr)
                err = 1.0 - accuracy_score(y_val, fitted.predict(C_val[:, :d]))
                if err < best_err:
                    best_err, best_d, best_name, best_clf = err, d, name, fitted
            except Exception:
                continue   # W-QDA raises when d exceeds class sample count

    plot_wavelet_cluster_view(
        np.vstack([C_tr, C_val]), np.concatenate([y_tr, y_val]), best_d, f"{DATASET} — {domain}"
    )

    y_pred = best_clf.predict(C_te[:, :best_d])
    print(
        f"{domain:9s}| {best_name:6s}| d={best_d:2d} | val_err={best_err:.3f} | "
        f"acc={accuracy_score(y_test, y_pred):.3f} | "
        f"prec={precision_score(y_test, y_pred, **METRIC_KW):.3f} | "
        f"rec={recall_score(y_test, y_pred, **METRIC_KW):.3f} | "
        f"f1={f1_score(y_test, y_pred, **METRIC_KW):.3f}"
    )


print(f"Dataset: {DATASET}")
print("domain   | model  | d    | val error | test metrics")
for dom in (["raw", "spectrum"] if DOMAIN == "both" else [DOMAIN]):
    run(dom)

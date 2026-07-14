"""
Classification pipeline — Berlinet, Biau & Rouvière, Section 2.

Steps
-----
1. Compute DWT coefficient matrix for all signals.
2. Rank basis functions by empirical energy on the training set  (eq. 2.4).
3. For each dimension d and each classifier, train on the first d
   reordered training coefficients.
4. Select the best (d, classifier) by minimising the empirical error
   on the validation set  (eq. 2.5).
5. Report classification metrics on the held-out test set.
"""

import numpy as np
import pywt
import warnings

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from functional_supervised_classification.data_loading import load_ecg200 as load_data

warnings.filterwarnings("ignore")  # ignore pywt warnings
# ── Parameters ────────────────────────────────────────────────────────────────

WAVELET  = "db4"
J        = 6
D_MAX    = 50    # upper bound on dimension search (W-QDA fails for large d)
VAL_FRAC = 0.3   # fraction of training data reserved for validation

CLASSIFIERS = {
    "W-NN"  : KNeighborsClassifier(),
    "W-QDA" : QuadraticDiscriminantAnalysis(),
    "W-CART": DecisionTreeClassifier(),
}

# ── Data loading and train/validation split ───────────────────────────────────

X_train_raw, y_train, X_test_raw, y_test = load_data()

# aeon: (n_samples, n_channels, n_timepoints) — squeeze channel dimension
S_train = X_train_raw[:, 0, :]
S_test  = X_test_raw[:, 0, :]

S_tr, S_val, y_tr, y_val = train_test_split(
    S_train, y_train, test_size=VAL_FRAC, random_state=42, stratify=y_train
)

# ── DWT coefficient matrices ──────────────────────────────────────────────────

def coeff_matrix(signals: np.ndarray) -> np.ndarray:
    """Compute DWT coefficients for every signal; returns shape (n, N_coeffs)."""
    return np.stack([
        np.concatenate(pywt.wavedec(s, WAVELET, level=J)) for s in signals
    ])

C_tr  = coeff_matrix(S_tr)
C_val = coeff_matrix(S_val)
C_te  = coeff_matrix(S_test)

# ── Energy-based ranking  (eq. 2.4) ──────────────────────────────────────────

ranking = np.argsort(np.sum(C_tr ** 2, axis=0))[::-1]

C_tr  = C_tr[:, ranking]
C_val = C_val[:, ranking]
C_te  = C_te[:, ranking]

# ── Joint selection of d and classifier  (eq. 2.5) ───────────────────────────

best_err, best_d, best_name, best_clf = np.inf, None, None, None

for d in range(1, min(D_MAX, C_tr.shape[1]) + 1):
    for name, clf in CLASSIFIERS.items():
        try:
            fitted = clone(clf)
            fitted.fit(C_tr[:, :d], y_tr)
            err = 1.0 - accuracy_score(y_val, fitted.predict(C_val[:, :d]))
            if err < best_err:
                best_err, best_d, best_name, best_clf = err, d, name, fitted
        except Exception:
            continue   # W-QDA raises when d exceeds class sample count

# ── Test metrics ──────────────────────────────────────────────────────────────

y_pred = best_clf.predict(C_te[:, :best_d])

kw = dict(average="binary", pos_label="1", zero_division=0)
print(f"Selected : {best_name}  |  d = {best_d}  |  val error = {best_err:.3f}")
print(f"Accuracy : {accuracy_score(y_test,  y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, **kw):.3f}")
print(f"Recall   : {recall_score(y_test,    y_pred, **kw):.3f}")
print(f"F1-score : {f1_score(y_test,        y_pred, **kw):.3f}")

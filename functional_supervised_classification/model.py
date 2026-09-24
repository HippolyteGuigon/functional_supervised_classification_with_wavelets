"""
Classification pipeline — Berlinet, Biau & Rouvière, Section 2.

Steps
-----
1. Optionally map each raw signal to its periodogram  (feature engineering).
2. Compute DWT coefficient matrix for all signals.
3. Build the *common representation*: rank the basis functions and keep the
   first d.  Two rules are available (``selection_rule`` in configs/config.yaml):
     - ``energy``             : empirical energy  sum_i X_ij^2            (eq. 2.4)
     - ``usage``              : usage frequency  phi_ij  of each feature  (eq. 2.4')
     - ``usage_discriminant`` : cross-class gap of the usage frequency
4. Jointly select the significance threshold, the dimension d and the classifier
   by K-fold cross-validation on the training set  (generalises eq. 2.5).  The
   ranking is recomputed inside every fold, from that fold's training part only.
5. Refit the winning (threshold, d, classifier) on the whole training set and
   report classification metrics on the held-out test set.
6. Plot a 2D projection of the selected coefficients, coloured by class.

The input representation is chosen in configs/config.yaml via ``input_domain``:
``raw`` (time domain), ``spectrum`` (periodogram) or ``both`` to compare them.
Datasets listed in ``SPECTRAL_DATASETS`` (phoneme) are already log-periodograms:
their only domain is ``spectrum`` and the data are used as provided, without
applying ``to_periodogram`` a second time.
"""

import warnings
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from functional_supervised_classification.coeffient_compute import coeff_matrix, to_periodogram
from functional_supervised_classification.config import load_config
from functional_supervised_classification.data_loading import load_ecg200, load_phoneme
from functional_supervised_classification.data_visualisation import plot_wavelet_cluster_view
from functional_supervised_classification.representation import (
    SELECTION_RULES,
    SIGNIFICANCE_RULES,
    feature_ranking,
    selection_stability,
)

warnings.filterwarnings("ignore")

CFG     = load_config()
DATASET = CFG["dataset"]
DOMAIN  = CFG.get("input_domain", "raw")

SELECTION_RULE = CFG.get("selection_rule", "energy")
_SIG           = CFG.get("significance") or {}
SIG_RULE       = _SIG.get("rule", "global")
SIG_PARAM_CFG  = _SIG.get("param", None)          # None -> cross-validate a grid
NORMALIZE      = bool(CFG.get("normalize", True)) # usage rules: unit-energy curves
CV_FOLDS       = int(CFG.get("cv_folds", 5))
D_MAX          = int(CFG.get("d_max", 50))        # W-QDA fails for large d

assert DATASET in ["phoneme", "ecg200"], f"Dataset {DATASET} not supported. Choose 'phoneme' or 'ecg200'."
assert DOMAIN in ["raw", "spectrum", "both"], f"input_domain {DOMAIN} not supported. Choose 'raw', 'spectrum' or 'both'."
assert SELECTION_RULE in SELECTION_RULES, f"selection_rule {SELECTION_RULE} not in {SELECTION_RULES}."
assert SIG_RULE in SIGNIFICANCE_RULES, f"significance.rule {SIG_RULE} not in {SIGNIFICANCE_RULES}."

# datasets whose curves are already (log-)periodograms
SPECTRAL_DATASETS = {"phoneme"}

# ── Parameters ────────────────────────────────────────────────────────────────

CLASSIFIERS = {
    "W-NN"  : KNeighborsClassifier(),
    "W-QDA" : QuadraticDiscriminantAnalysis(),
    "W-CART": DecisionTreeClassifier(),
    "W-FFNN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
}

# feature engineering applied to the raw signals before the wavelet decomposition
TRANSFORMS = {"raw": lambda s: s, "spectrum": to_periodogram}

# grid searched by cross-validation for the significance threshold of the
# "usage" rules; "energy" ignores the threshold, hence a single ``None``.
PARAM_GRID = {
    "energy"   : [None],
    "global"   : [0.5, 1.0, 1.5, 2.0, 3.0],
    "top_m"    : [8, 16, 32, 64],
    "quantile" : [0.80, 0.90, 0.95],
    "universal": [0.7, 1.0, 1.3],
}

# ── Data loading ─────────────────────────────────────────────────────────────

X_train_raw, y_train, X_test_raw, y_test = (load_phoneme if DATASET == "phoneme" else load_ecg200)()

S_train = X_train_raw[:, 0, :]
S_test  = X_test_raw[:, 0, :]

if len(np.unique(y_test)) == 2:
    METRIC_KW = dict(average="binary", pos_label=sorted(np.unique(y_test))[-1], zero_division=0)
else:
    METRIC_KW = dict(average="macro", zero_division=0)


def _param_grid(n_features: int) -> list:
    """Threshold values to cross-validate, given the config and the coefficient count."""
    if SELECTION_RULE == "energy":
        return [None]
    if SIG_PARAM_CFG is not None:
        return [SIG_PARAM_CFG]
    grid = PARAM_GRID[SIG_RULE]
    if SIG_RULE == "top_m":                       # drop values clamped to the same cap
        grid = sorted({min(int(m), n_features) for m in grid})
    return grid


def run(domain: str) -> None:
    """Run the full pipeline (steps 2–6) on the chosen input representation.

    Arguments:
        domain (str): The input domain to use ('raw' or 'spectrum').
    Returns:
        None
    """
    # spectral datasets are used as provided: no second periodogram
    tf = (lambda s: s) if DATASET in SPECTRAL_DATASETS else TRANSFORMS[domain]
    C_train = coeff_matrix(tf(S_train))
    C_test  = coeff_matrix(tf(S_test))
    d_hi    = min(D_MAX, C_train.shape[1])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(C_train, y_train))

    # joint cross-validated search over (threshold, d, classifier)
    best = {"err": np.inf, "param": None, "d": None, "name": None}
    for param in _param_grid(C_train.shape[1]):
        # rank + reorder once per fold for this threshold (no leakage: the
        # ranking sees only the fold's training rows)
        folds = []
        for tr_idx, va_idx in splits:
            ranking = feature_ranking(
                C_train[tr_idx], y_train[tr_idx],
                selection_rule=SELECTION_RULE,
                significance_rule=SIG_RULE, significance_param=param,
                normalize=NORMALIZE,
            )
            folds.append((
                C_train[tr_idx][:, ranking], y_train[tr_idx],
                C_train[va_idx][:, ranking], y_train[va_idx],
            ))

        for d in range(1, d_hi + 1):
            for name, clf in CLASSIFIERS.items():
                errs = []
                for C_tr, y_tr, C_va, y_va in folds:
                    try:
                        fitted = clone(clf).fit(C_tr[:, :d], y_tr)
                        errs.append(1.0 - accuracy_score(y_va, fitted.predict(C_va[:, :d])))
                    except Exception:
                        break                    # W-QDA raises when d exceeds class sample count
                if len(errs) == len(folds):
                    mean_err = float(np.mean(errs))
                    if mean_err < best["err"]:
                        best = {"err": mean_err, "param": param, "d": d, "name": name}

    # refit the winner on the whole training set
    full_ranking = feature_ranking(
        C_train, y_train,
        selection_rule=SELECTION_RULE,
        significance_rule=SIG_RULE, significance_param=best["param"],
        normalize=NORMALIZE,
    )
    C_tr_full = C_train[:, full_ranking]
    C_te_full = C_test[:, full_ranking]
    final = clone(CLASSIFIERS[best["name"]]).fit(C_tr_full[:, :best["d"]], y_train)
    y_pred = final.predict(C_te_full[:, :best["d"]])

    # stability of the selected index set across the folds
    fold_rankings = [
        feature_ranking(
            C_train[tr_idx], y_train[tr_idx],
            selection_rule=SELECTION_RULE,
            significance_rule=SIG_RULE, significance_param=best["param"],
            normalize=NORMALIZE,
        )
        for tr_idx, _ in splits
    ]
    stab = selection_stability(fold_rankings, best["d"])

    plot_wavelet_cluster_view(C_tr_full, y_train, best["d"], f"{DATASET} — {domain}")

    param_str = "-" if best["param"] is None else f"{best['param']:g}"
    print(
        f"{domain:9s}| {best['name']:6s}| d={best['d']:2d} | thr={param_str:>5s} | "
        f"cv_err={best['err']:.3f} | jaccard={stab:.2f} | "
        f"acc={accuracy_score(y_test, y_pred):.3f} | "
        f"prec={precision_score(y_test, y_pred, **METRIC_KW):.3f} | "
        f"rec={recall_score(y_test, y_pred, **METRIC_KW):.3f} | "
        f"f1={f1_score(y_test, y_pred, **METRIC_KW):.3f}"
    )


if __name__ == "__main__":
    print(f"Dataset: {DATASET}   |   representation rule: {SELECTION_RULE}"
          f"   |   significance: {SIG_RULE}   |   CV folds: {CV_FOLDS}")
    print("domain   | model  | d    | thr   | cv error | stability | test metrics")
    if DATASET in SPECTRAL_DATASETS:
        if DOMAIN != "spectrum":
            print(f"{DATASET} curves are already log-periodograms: running the "
                  f"'spectrum' domain on the data as provided.")
        domains = ["spectrum"]
    else:
        domains = ["raw", "spectrum"] if DOMAIN == "both" else [DOMAIN]
    for dom in domains:
        run(dom)

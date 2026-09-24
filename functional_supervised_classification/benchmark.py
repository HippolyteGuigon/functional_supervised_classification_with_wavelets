"""
Repeated benchmark with the data-splitting protocol of Berlinet, Biau & Rouvière.

This is the protocol covered by the theory (Section 1.2 and eq. 2.5 of the
paper, and the consistency theorem of the thesis): for every repetition,

1. draw a random stratified partition  training (n) / validation (m) / test;
2. compute the wavelet coefficients (orthogonal periodized DWT) and rank the
   basis functions with the training sequence only, by energy (eq. 2.4) or by
   usage frequency with a global threshold on unit-energy curves (eq. 2.4');
3. for every threshold c, dimension d and rule g of the family, train g on the
   first d reordered coefficients of the training sequence and compute its
   error on the validation sequence;
4. keep the minimiser of the validation error (ties: smallest d) and report its
   error on the test sequence.

Selection is done within each classifier family (W-NN, W-QDA, W-CART, as in
Tables 1 and 3 of the paper) and jointly over the three families ("ALL").

Usage::

    python -m functional_supervised_classification.benchmark phoneme
    python -m functional_supervised_classification.benchmark all --reps 50

Results are appended as CSV files to ``results/``.
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from functional_supervised_classification.coeffient_compute import coeff_matrix, to_periodogram
from functional_supervised_classification.representation import feature_ranking
from functional_supervised_classification.simulation import contaminate, simulate_curves

warnings.filterwarnings("ignore")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

C_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)               # tau = c / sqrt(p)
K_GRID = (1, 3, 5, 7, 9, 15, 25)                 # neighbours of W-NN
FAMILIES = ("W-NN", "W-QDA", "W-CART")
TOP_STORE = 20                                   # ranked indices kept per run


# ── data ─────────────────────────────────────────────────────────────────────

def _load_phoneme_all():
    df = pd.read_csv("https://hastie.su.domains/ElemStatLearn/datasets/phoneme.data")
    X = df[[c for c in df.columns if c.startswith("x.")]].to_numpy(dtype=float)
    return X, df["g"].to_numpy()


def _load_ecg200_all():
    from functional_supervised_classification.data_loading import load_ecg200
    Xa, ya, Xb, yb = load_ecg200()
    return np.vstack([Xa[:, 0, :], Xb[:, 0, :]]).astype(float), np.concatenate([ya, yb])


# every setting: data loader (None = simulated), feature map, sizes (n, m, test)
SETTINGS = {
    # phoneme curves are already log-periodograms: used as provided
    "phoneme":        dict(load=_load_phoneme_all, transform=None, n=250, m=250, test=None),
    "ecg200-raw":     dict(load=_load_ecg200_all, transform=None, n=50, m=50, test=None),
    "ecg200-spectrum": dict(load=_load_ecg200_all, transform=to_periodogram, n=50, m=50, test=None),
    "simulation":     dict(load=None, transform=None, n=50, m=50, test=500),
}


# ── one run ──────────────────────────────────────────────────────────────────

def _candidates(n_train: int):
    """(family, label, estimator factory) for every rule of the collection D_n^(d)."""
    out = [("W-NN", f"k={k}", lambda k=k: KNeighborsClassifier(n_neighbors=k))
           for k in K_GRID if k <= n_train]
    out.append(("W-QDA", "qda", lambda: QuadraticDiscriminantAnalysis()))
    out.append(("W-CART", "cart", lambda: DecisionTreeClassifier(random_state=0)))
    return out


def _qda_d_max(y: np.ndarray) -> int:
    """QDA needs more training points than dimensions in every class (paper, Sec. 3)."""
    return int(min(np.unique(y, return_counts=True)[1])) - 1


def run_once(C_tr, y_tr, C_va, y_va, C_te, y_te, rule: str, d_max: int) -> list[dict]:
    """
    Select (c, d, g) on the validation sequence and evaluate on the test one.

    Returns one record per family plus one for the joint selection ("ALL").
    """
    params = [None] if rule == "energy" else list(C_GRID)
    cands = _candidates(len(y_tr))
    qda_max = _qda_d_max(y_tr)
    rows = []                                            # (family, err, d, c, label, g_idx)
    rankings = {}
    for c in params:
        ranking = feature_ranking(C_tr, selection_rule="energy" if rule == "energy" else "usage",
                                  significance_rule="global", significance_param=c)
        rankings[c] = ranking
        A_tr, A_va = C_tr[:, ranking], C_va[:, ranking]
        for d in range(1, d_max + 1):
            for g_idx, (fam, label, make) in enumerate(cands):
                if fam == "W-QDA" and d > qda_max:
                    continue
                try:
                    clf = make().fit(A_tr[:, :d], y_tr)
                    err = float(np.mean(clf.predict(A_va[:, :d]) != y_va))
                except Exception:                        # singular covariance in QDA
                    continue
                rows.append((fam, err, d, c, label, g_idx))

    records = []
    for fam in FAMILIES + ("ALL",):
        pool = [r for r in rows if fam == "ALL" or r[0] == fam]
        if not pool:
            continue
        best = min(pool, key=lambda r: (r[1], r[2]))     # smallest error, then smallest d
        g_fam, val_err, d, c, label, g_idx = best
        ranking = rankings[c]
        clf = cands[g_idx][2]().fit(C_tr[:, ranking][:, :d], y_tr)
        test_err = float(np.mean(clf.predict(C_te[:, ranking][:, :d]) != y_te))
        records.append(dict(
            family=fam, chosen=f"{g_fam}:{label}", d=d,
            c=np.nan if c is None else c, val_err=val_err, test_err=test_err,
            top=" ".join(map(str, ranking[:TOP_STORE])),
            top_c1=" ".join(map(str, rankings.get(1.0, ranking)[:TOP_STORE])),
        ))
    return records


def _split(X, y, n, m, test, seed):
    idx = np.arange(len(y))
    i_tr, i_rest = train_test_split(idx, train_size=n, stratify=y, random_state=seed)
    i_va, i_te = train_test_split(i_rest, train_size=m, stratify=y[i_rest], random_state=seed)
    if test is not None:
        i_te = i_te[:test]
    return i_tr, i_va, i_te


def _one_repetition(setting: str, X, y, rep: int, wavelet: str, rules, d_max: int,
                    n: int, m: int, n_outliers: int = 0) -> list[dict]:
    cfg = SETTINGS[setting]
    seed = 1000 + rep
    rng = np.random.default_rng(seed)
    if cfg["load"] is None:                                   # fresh simulated sample
        test = cfg["test"]
        X, y = simulate_curves(n + m + test, rng)
        i_tr, i_va, i_te = np.arange(n), np.arange(n, n + m), np.arange(n + m, n + m + test)
    else:
        i_tr, i_va, i_te = _split(X, y, n, m, cfg["test"], seed)
    X_tr = X[i_tr]
    if n_outliers:
        X_tr, _ = contaminate(X_tr, n_outliers, rng)
    tf = cfg["transform"] or (lambda s: s)
    C_tr = coeff_matrix(tf(X_tr), wavelet=wavelet)
    C_va = coeff_matrix(tf(X[i_va]), wavelet=wavelet)
    C_te = coeff_matrix(tf(X[i_te]), wavelet=wavelet)
    d_hi = min(d_max, C_tr.shape[1])
    out = []
    for rule in rules:
        for rec in run_once(C_tr, y[i_tr], C_va, y[i_va], C_te, y[i_te], rule, d_hi):
            rec.update(setting=setting, wavelet=wavelet, rule=rule, rep=rep,
                       n=n, m=m, n_outliers=n_outliers)
            out.append(rec)
    return out


def benchmark(setting: str, reps: int, wavelets=("db4",), rules=("energy", "usage"),
              d_max: int = 64, n: int | None = None, m: int | None = None,
              outliers=(0,), n_jobs: int = -1) -> pd.DataFrame:
    """Run ``reps`` random partitions for every (wavelet, contamination level)."""
    cfg = SETTINGS[setting]
    n = n or cfg["n"]
    m = m or cfg["m"]
    X, y = cfg["load"]() if cfg["load"] is not None else (None, None)
    jobs = [delayed(_one_repetition)(setting, X, y, r, w, rules, d_max, n, m, k)
            for w in wavelets for k in outliers for r in range(reps)]
    res = Parallel(n_jobs=n_jobs)(jobs)
    return pd.DataFrame([rec for block in res for rec in block])


# ── experiments reported in the thesis ───────────────────────────────────────

EXPERIMENTS = {
    # Tables 1 and 3 of the paper, energy vs usage, six Daubechies bases
    "phoneme":    lambda R: benchmark("phoneme", R, wavelets=("db1", "db2", "db4", "db6", "db8", "db10")),
    "simulation": lambda R: benchmark("simulation", R, wavelets=("db1", "db2", "db4", "db6", "db8", "db10")),
    # time domain vs periodogram on a genuine time series
    "ecg200":     lambda R: pd.concat([benchmark("ecg200-raw", R), benchmark("ecg200-spectrum", R)]),
    # robustness: spikes added to a few training curves
    "contamination": lambda R: benchmark("simulation", R, wavelets=("db4",), outliers=(0, 1, 3, 5)),
    # consistency: error as the sample size grows (n = m)
    "sample_size": lambda R: pd.concat([
        benchmark("simulation", R, wavelets=("db4",), n=s, m=s) for s in (25, 50, 100, 200, 400)
    ]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--reps", type=int, default=50)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for name in names:
        t0 = time.time()
        df = EXPERIMENTS[name](args.reps)
        df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows in {time.time() - t0:.0f}s -> results/{name}.csv")


if __name__ == "__main__":
    main()

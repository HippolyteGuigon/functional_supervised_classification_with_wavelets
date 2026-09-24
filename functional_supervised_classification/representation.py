"""
Step 2 (revised) — the "common representation" selection rule.

Berlinet, Biau & Rouvière define the common representation of every functional
observation as the projection onto a *single* subset ``S_d`` of wavelet basis
functions, shared by all curves:

    Phi : x  |->  ( <x, psi_{j,k}> )_{(j,k) in S_d}   in  R^d

Their rule (eq. 2.4) chooses ``S_d`` by ranking the basis functions ``(j, k)``
by empirical energy on the training sample

    E_{j,k} = sum_i X_{i,j,k}^2

and keeping the ``d`` most energetic ones.  Energy is not robust: a single curve
with a large coefficient in ``(j, k)`` is enough to pull that index into the
"common" representation, even if no other curve uses it.

This module implements the alternative rule asked for by the thesis supervisor:
rank the time-frequency features ``(j, k)`` by *how frequently they are used*
across the sample, i.e. how often the corresponding coefficient is significant
in an individual signal.

    s_{i,j,k} = 1[ |X~_{i,j,k}| > tau ]         significance indicator
    phi_{j,k} = (1 / n) sum_i s_{i,j,k}         usage frequency        (eq. 2.4')
    S_d       = the d indices of largest phi_{j,k}

where ``X~_i = X_i / ||X_i||`` is the curve rescaled to unit energy, so that a
single global threshold ``tau`` means the same thing for every curve. The
default rule (``"global"``, retained with the thesis supervisor) takes
``tau = c / sqrt(p)``: a coefficient is "used" by a curve when it carries more
than ``c^2`` times the share of energy it would get if the curve's energy were
spread evenly over the ``p`` coefficients.

A class-conditional variant ranks by how *differently* a feature is used from
one class to another, targeting discriminative power directly.

The significance threshold and the retained dimension ``d`` are tuned by
cross-validation in :mod:`functional_supervised_classification.model`.
"""

import numpy as np

SELECTION_RULES = ("energy", "usage", "usage_discriminant")
SIGNIFICANCE_RULES = ("global", "top_m", "quantile", "universal")

# 1 / Phi^{-1}(0.75): converts a median-absolute-deviation into a Gaussian sigma.
_MAD_TO_SIGMA = 1.0 / 0.6744897501960817

_DEFAULT_PARAM = {"global": 1.0, "top_m": 20, "quantile": 0.9, "universal": 1.0}


def normalize_rows(C: np.ndarray) -> np.ndarray:
    """
    Scale every curve to unit norm: ``X_i / ||X_i||``.

    For an orthogonal DWT (see :func:`coeff_matrix`) normalising the coefficient
    vector is the same as normalising the discretised curve in ``L^2``, so all
    curves share the same total energy ``sum_j X_ij^2 = 1``. A global
    significance threshold then means the same thing for every curve. Null rows
    are left unchanged.
    """
    C = np.asarray(C, dtype=float)
    norms = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.where(norms > 0.0, norms, 1.0)


def significance_mask(C: np.ndarray, rule: str = "top_m", param=None) -> np.ndarray:
    """
    Boolean matrix ``s_{i,j,k}`` flagging, for each signal, the coefficients it
    "uses" (whose magnitude is significant).

    Parameters
    ----------
    C:
        DWT coefficient matrix, shape ``(n, p)`` (one row per signal), as
        returned by :func:`coeffient_compute.coeff_matrix`.
    rule:
        - ``"global"``    : one threshold ``tau = param / sqrt(p)`` shared by all
          curves (``param`` a positive multiplier, default 1.0). Meant for
          rows normalised by :func:`normalize_rows`.
        - ``"top_m"``     : the ``param`` largest-magnitude coefficients of each
          signal are flagged (``param`` an int, default 20).
        - ``"quantile"``  : coefficients strictly above the global ``param``
          quantile of ``|C|`` are flagged (``param`` in ``(0, 1)``, default 0.9).
        - ``"universal"`` : Donoho-Johnstone universal threshold
          ``param * sigma_hat * sqrt(2 ln p)`` with ``sigma_hat`` a MAD-based
          robust noise scale of ``C`` (``param`` a multiplier, default 1.0).
    param:
        Rule parameter; ``None`` falls back to the per-rule default above.

    Returns
    -------
    np.ndarray of bool, shape ``(n, p)``.
    """
    C = np.asarray(C, dtype=float)
    if C.ndim != 2:
        raise ValueError(f"C must be 2-D (n, p); got shape {C.shape}")
    if rule not in SIGNIFICANCE_RULES:
        raise ValueError(f"unknown significance rule {rule!r}; choose from {SIGNIFICANCE_RULES}")

    n, p = C.shape
    A = np.abs(C)
    if param is None:
        param = _DEFAULT_PARAM[rule]

    if rule == "global":
        c = float(param)
        if c <= 0.0:
            raise ValueError(f"global parameter must be > 0; got {c}")
        return A > c / np.sqrt(p)

    if rule == "top_m":
        m = int(param)
        if m < 1:
            raise ValueError(f"top_m parameter must be >= 1; got {m}")
        m = min(m, p)
        mask = np.zeros((n, p), dtype=bool)
        top = np.argsort(A, axis=1)[:, p - m:]          # m largest |coef| per row
        np.put_along_axis(mask, top, True, axis=1)
        return mask

    if rule == "quantile":
        q = float(param)
        if not 0.0 < q < 1.0:
            raise ValueError(f"quantile parameter must be in (0, 1); got {q}")
        return A > np.quantile(A, q)

    # rule == "universal"
    c = float(param)
    sigma = _MAD_TO_SIGMA * np.median(A)
    if not np.isfinite(sigma) or sigma == 0.0:
        sigma = float(A.std()) or 1.0
    tau = c * sigma * np.sqrt(2.0 * np.log(max(p, 2)))
    return A > tau


def usage_frequency(mask: np.ndarray) -> np.ndarray:
    """``phi_{j,k}`` in ``[0, 1]``: fraction of signals that use each feature (eq. 2.4')."""
    return np.asarray(mask, dtype=bool).mean(axis=0)


def usage_frequency_by_class(mask: np.ndarray, y) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-class usage frequency.

    Returns
    -------
    classes:
        Sorted unique labels, shape ``(n_classes,)``.
    freqs:
        ``freqs[c]`` is ``phi^{(class c)}_{j,k}``, shape ``(n_classes, p)``.
    """
    mask = np.asarray(mask, dtype=bool)
    y = np.asarray(y)
    classes = np.unique(y)
    freqs = np.stack([mask[y == c].mean(axis=0) for c in classes])
    return classes, freqs


def discriminative_usage(mask: np.ndarray, y) -> np.ndarray:
    """
    Score each feature by how differently it is used across classes.

    Binary case      : ``|phi^{(1)} - phi^{(0)}|``.
    Multiclass case  : population variance of ``phi^{(c)}`` across classes. For
    two classes this variance equals ``((phi^{(1)} - phi^{(0)}) / 2) ** 2``,
    monotone in the binary score, so the two produce the same ranking.
    """
    classes, freqs = usage_frequency_by_class(mask, y)
    if len(classes) == 2:
        return np.abs(freqs[1] - freqs[0])
    return freqs.var(axis=0)


def energy(C: np.ndarray) -> np.ndarray:
    """``E_{j,k} = sum_i X_{i,j,k}^2`` — the original rule (eq. 2.4)."""
    C = np.asarray(C, dtype=float)
    return np.sum(C ** 2, axis=0)


def feature_ranking(
    C: np.ndarray,
    y=None,
    selection_rule: str = "usage",
    significance_rule: str = "global",
    significance_param=None,
    normalize: bool = True,
) -> np.ndarray:
    """
    Order the columns of ``C`` (the wavelet basis functions) from most to least
    informative, according to the chosen common-representation rule.

    Parameters
    ----------
    C:
        DWT coefficient matrix, shape ``(n, p)``.
    y:
        Class labels, shape ``(n,)``; required only for
        ``selection_rule="usage_discriminant"``.
    selection_rule:
        - ``"energy"``             : original eq. 2.4 (ignores ``significance_*``).
        - ``"usage"``              : rank by ``phi_{j,k}``            (eq. 2.4').
        - ``"usage_discriminant"`` : rank by the cross-class usage gap.
    significance_rule, significance_param:
        Passed to :func:`significance_mask` for the two "usage" rules.
    normalize:
        For the two "usage" rules, rescale every curve to unit energy
        (:func:`normalize_rows`) before thresholding. Only the ranking uses the
        normalised curves; the classifier still receives the original
        coefficients. Ignored by ``"energy"``, which follows eq. 2.4 as is.

    Returns
    -------
    np.ndarray of int, shape ``(p,)``
        Column indices, ``ranking[0]`` the most informative. Ties keep the
        natural (ascending index) order.
    """
    if selection_rule not in SELECTION_RULES:
        raise ValueError(
            f"unknown selection_rule {selection_rule!r}; choose from {SELECTION_RULES}"
        )

    if selection_rule == "energy":
        score = energy(C)
    else:
        C_sel = normalize_rows(C) if normalize else C
        mask = significance_mask(C_sel, rule=significance_rule, param=significance_param)
        if selection_rule == "usage":
            score = usage_frequency(mask)
        else:  # usage_discriminant
            if y is None:
                raise ValueError("selection_rule='usage_discriminant' requires y")
            score = discriminative_usage(mask, y)

    return np.argsort(-score, kind="stable")


def selection_stability(rankings, d: int) -> float:
    """
    Mean pairwise Jaccard overlap of the top-``d`` index sets across a list of
    rankings (e.g. one per cross-validation fold). ``1.0`` = identical
    selections, ``0.0`` = disjoint. Returns ``nan`` for fewer than two rankings.
    """
    sets = [set(np.asarray(r)[:d].tolist()) for r in rankings]
    if len(sets) < 2:
        return float("nan")
    scores = [
        len(a & b) / len(a | b)
        for i, a in enumerate(sets)
        for b in sets[i + 1:]
    ]
    return float(np.mean(scores))

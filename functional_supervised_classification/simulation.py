"""
Simulated curves of Berlinet, Biau & Rouvière, Section 3.2, and the outlier
contamination used to compare the robustness of the common-representation rules.

Each curve is made of two windowed sinusoids placed symmetrically around 1/2,

    X(t) = sin(F1 pi t) f_{mu, sigma}(t) + sin(F2 pi t) f_{1 - mu, sigma}(t) + eps_t

with F1, F2 ~ U[50, 150], mu ~ U[0.1, 0.4], sigma^2 ~ U[0, 0.005], eps_t iid
N(0, 0.5^2), sampled at T = 1024 equidistant points of [0, 1]. The label is
Y = 0 if mu <= 0.25 (the two bursts are far apart) and Y = 1 otherwise.
"""

import numpy as np

T_SIM = 1024
_SIGMA2_FLOOR = 1e-6          # sigma^2 = 0 would make the Gaussian window a Dirac


def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


def simulate_curves(n: int, rng: np.random.Generator, T: int = T_SIM,
                    noise_sd: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw ``n`` labelled curves from the model of Section 3.2.

    Returns
    -------
    X : np.ndarray, shape (n, T)
    y : np.ndarray of int in {0, 1}, shape (n,)
    """
    t = np.arange(T) / T
    F1 = rng.uniform(50.0, 150.0, size=n)
    F2 = rng.uniform(50.0, 150.0, size=n)
    mu = rng.uniform(0.1, 0.4, size=n)
    sigma = np.sqrt(np.maximum(rng.uniform(0.0, 0.005, size=n), _SIGMA2_FLOOR))
    X = np.empty((n, T))
    for i in range(n):
        X[i] = (np.sin(F1[i] * np.pi * t) * _gauss(t, mu[i], sigma[i])
                + np.sin(F2[i] * np.pi * t) * _gauss(t, 1.0 - mu[i], sigma[i]))
    X += rng.normal(scale=noise_sd, size=(n, T))
    y = (mu > 0.25).astype(int)
    return X, y


def contaminate(X: np.ndarray, n_outliers: int, rng: np.random.Generator,
                height: float = 50.0, width: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Add a narrow spike to ``n_outliers`` randomly chosen curves.

    The spike is a Gaussian bump of standard deviation ``width`` samples, placed
    at a uniform random position, whose height is ``height`` times the largest
    absolute value of the clean sample. Labels are left untouched: only the
    curves change, which models recording artefacts.

    Returns
    -------
    Xc : np.ndarray, the contaminated copy of ``X``
    idx : np.ndarray of int, the indices of the contaminated curves
    """
    Xc = np.array(X, dtype=float, copy=True)
    n, T = Xc.shape
    idx = rng.choice(n, size=min(n_outliers, n), replace=False)
    grid = np.arange(T)
    amp = height * np.max(np.abs(X))
    for i in idx:
        t0 = rng.integers(0, T)
        Xc[i] += amp * np.exp(-0.5 * ((grid - t0) / width) ** 2)
    return Xc, idx

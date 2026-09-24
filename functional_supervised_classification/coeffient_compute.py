"""
Step 1 — Wavelet coefficient computation for the classification pipeline.

Goal
----
For each signal X_i, compute the full DWT coefficient vector

    X_i = (X_i1, X_i2, ..., X_i{2^J})

where X_ij = <X_i, psi_j> = int_0^1 X_i(t) psi_j(t) dt

as defined in eq. (2.3) of Berlinet, Biau & Rouvière.

This vector is the input to the two subsequent steps of the pipeline:
  - energy-based ranking of basis functions  (eq. 2.4)
  - joint selection of dimension d and classifier g  (eq. 2.5)
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import detrend as _detrend, periodogram

from functional_supervised_classification.data_loading import load_ecg200 as load_data


WAVELET = "db4"
J       = None          # None -> deepest dyadic level allowed by the signal length
MODE    = "periodization"


def dyadic_level(T: int) -> int:
    """
    Largest ``L`` such that ``2^L`` divides ``T``: the deepest level at which the
    periodized DWT stays an orthogonal change of basis of ``R^T``
    (``T = 96 = 2^5 * 3 -> 5``, ``T = 256 -> 8``, ``T = 1024 -> 10``).
    """
    if T < 2:
        raise ValueError(f"signal length must be >= 2; got {T}")
    L = 0
    while T % (2 ** (L + 1)) == 0:
        L += 1
    return L


def coeff_matrix(
    signals: np.ndarray,
    wavelet: str = WAVELET,
    level: int | None = J,
    mode: str = MODE,
) -> np.ndarray:
    """
    Compute the full DWT coefficient matrix for a set of signals.

    Applies pywt.wavedec which computes, for each scale j and translation k:

        X_ij = <X_i, psi_{j,k}>  with  psi_{j,k}(t) = 2^{j/2} psi(2^j t - k)

    and returns them concatenated as [cA_J, cD_J, cD_{J-1}, ..., cD_1],
    matching the reindexed basis {phi_{0,0}, psi_{0,0}, psi_{1,0}, ...}
    from eq. (2.3) of the paper.

    With ``mode="periodization"`` and a dyadic level (the defaults), the
    transform is an orthogonal matrix of ``R^T``: there are exactly ``T``
    coefficients and Parseval holds, ``sum_j X_ij^2 = sum_t x_i(t)^2``. The
    PyWavelets default ``mode="symmetric"`` pads the signal and returns more
    than ``T`` coefficients whose energy exceeds that of the signal (by ~25 %
    for db4 on ECG200), which breaks the energy interpretation of eq. (2.4) and
    the normalisation used by the usage rule.

    Parameters
    ----------
    signals:
        Raw signals on a uniform grid, shape (n, T).
    wavelet:
        PyWavelets wavelet identifier.
    level:
        Number of decomposition levels; ``None`` uses :func:`dyadic_level`.
    mode:
        PyWavelets signal extension mode.

    Returns
    -------
    np.ndarray
        DWT coefficient matrix, shape (n, T) in periodization mode.
    """
    signals = np.asarray(signals, dtype=float)
    if level is None:
        level = dyadic_level(signals.shape[1])
    with warnings.catch_warnings():
        # periodization is exact at every dyadic level; PyWavelets still warns
        # that deep levels "experience boundary effects"
        warnings.filterwarnings("ignore", message="Level value of")
        return np.stack([
            np.concatenate(pywt.wavedec(s, wavelet, mode=mode, level=level)) for s in signals
        ])


def to_periodogram(
    signals: np.ndarray,
    detrend: str = "constant",
    log: bool = True,
) -> np.ndarray:
    """
    Feature engineering: replace each raw time-domain signal by its periodogram.

    The periodogram I(w_j) = |DFT(x)|^2 is the "sample spectral density" (Shumway
    & Stoffer, Time Series Analysis, sec. 4.3); it is a non-consistent estimator
    of the true spectral density. Running the wavelet pipeline on this
    representation instead of the raw signal mirrors Section 3.1 of Berlinet, Biau
    & Rouvière, where the phoneme data are already log-periodograms.

    Two conventions from sec. 4.3 are applied:
      - the series is detrended before the DFT, since trends leak into the low
        frequencies ("constant" removes the mean, "linear" also removes an affine
        trend);
      - the zero-frequency (DC) ordinate is dropped: after detrending it carries
        no information (I(0) = n * mean^2 ~ 0) and would otherwise dominate the
        log scale.

    Parameters
    ----------
    signals:
        Raw signals on a uniform grid, shape (n, T).
    detrend:
        "constant" (mean removal, default) or "linear" (remove an affine trend).
    log:
        Return log-periodograms (variance-stabilised, as in the phoneme setup).

    Returns
    -------
    np.ndarray
        Spectral representation, shape (n, T // 2).
    """
    x = _detrend(signals, axis=1, type=detrend)
    _, power = periodogram(x, axis=1, detrend=False)
    power = power[:, 1:]                       # drop the DC ordinate I(0)
    return np.log(power + 1e-12) if log else power


def explore():
    """
    The goal of this function is to explore the DWT coefficient matrix for the ECG200 dataset.
    It loads the dataset, computes the DWT coefficients, and visualizes the results.
    
    Args:
        None
    Returns:
        None
    """
    X_train, y_train, _, _ = load_data()

    signals  = X_train[:, 0, :]
    X_coeffs = coeff_matrix(signals)

    energy  = np.sum(X_coeffs ** 2, axis=0)
    ranking = np.argsort(energy)[::-1]

    print(f"Signal matrix     : {signals.shape}")
    print(f"Coefficient matrix: {X_coeffs.shape}")
    print(f"Top-5 coefficient indices by energy : {ranking[:5]}")
    print(f"Their empirical energies            : {energy[ranking[:5]].round(4)}")

    _ , axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(signals[0])
    axes[0].set_title(f"Raw ECG signal  (class = {y_train[0]})")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Amplitude")

    axes[1].stem(X_coeffs[0], markerfmt="C1o", linefmt="C1-", basefmt="k-")
    axes[1].set_title(
        r"DWT coefficient vector  $X_i = (X_{i1}, \ldots, X_{i,2^J})$"
        f"  —  wavelet: {WAVELET}, level J={dyadic_level(signals.shape[1])}"
    )
    axes[1].set_xlabel("Coefficient index $j$")
    axes[1].set_ylabel("$X_{ij}$")

    axes[2].bar(range(len(energy)), energy[ranking], color="steelblue")
    axes[2].set_title(
        r"Empirical energy $\sum_{i=1}^n X_{ij}^2$ sorted by rank  (eq. 2.4)"
    )
    axes[2].set_xlabel("Rank (0 = most energetic)")
    axes[2].set_ylabel("Energy")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    explore()

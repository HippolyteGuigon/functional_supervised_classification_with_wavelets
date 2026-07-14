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

import numpy as np
import matplotlib.pyplot as plt
import pywt

from functional_supervised_classification.data_loading import load_ecg200 as load_data


WAVELET = "db4"
J       = 6


def coeff_matrix(
    signals: np.ndarray,
    wavelet: str = WAVELET,
    level: int = J,
) -> np.ndarray:
    """
    Compute the full DWT coefficient matrix for a set of signals.

    Applies pywt.wavedec which computes, for each scale j and translation k:

        X_ij = <X_i, psi_{j,k}>  with  psi_{j,k}(t) = 2^{j/2} psi(2^j t - k)

    and returns them concatenated as [cA_J, cD_J, cD_{J-1}, ..., cD_1],
    matching the reindexed basis {phi_{0,0}, psi_{0,0}, psi_{1,0}, ...}
    from eq. (2.3) of the paper.

    The signals are NOT normalized: coefficient magnitudes must preserve the
    true energy in each basis direction, as required by eq. (2.4).

    Parameters
    ----------
    signals:
        Raw signals on a uniform grid, shape (n, T).
    wavelet:
        PyWavelets wavelet identifier.
    level:
        Maximum resolution level J.

    Returns
    -------
    np.ndarray
        DWT coefficient matrix, shape (n, N_coeffs) with N_coeffs ≈ T.
    """
    return np.stack([
        np.concatenate(pywt.wavedec(s, wavelet, level=level)) for s in signals
    ])


if __name__ == "__main__":
    X_train, y_train, _, _ = load_data()

    # aeon returns shape (n_samples, n_channels, n_timepoints) — squeeze channel dim
    signals = X_train[:, 0, :]

    X_coeffs = coeff_matrix(signals)

    # Energy-based ranking (eq. 2.4)
    energy  = np.sum(X_coeffs ** 2, axis=0)
    ranking = np.argsort(energy)[::-1]

    print(f"Signal matrix     : {signals.shape}")
    print(f"Coefficient matrix: {X_coeffs.shape}")
    print(f"Top-5 coefficient indices by energy : {ranking[:5]}")
    print(f"Their empirical energies            : {energy[ranking[:5]].round(4)}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    axes[0].plot(signals[0])
    axes[0].set_title(f"Raw ECG signal  (class = {y_train[0]})")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Amplitude")

    axes[1].stem(X_coeffs[0], markerfmt="C1o", linefmt="C1-", basefmt="k-")
    axes[1].set_title(
        r"DWT coefficient vector  $X_i = (X_{i1}, \ldots, X_{i,2^J})$"
        f"  —  wavelet: {WAVELET}, level J={J}"
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

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


# ============================================================================
# Load ECG200
# ============================================================================

X_train, y_train, _, _ = load_data()

# aeon returns shape (n_samples, n_channels, n_timepoints).
# ECG200: (100, 1, 96) — squeeze the channel dimension.
signals = X_train[:, 0, :]   # shape (100, 96)


# ============================================================================
# Parameters
# ============================================================================

WAVELET = "db4"
J = 6        # resolution level — the paper truncates expansion at level J


# ============================================================================
# Wavelet coefficient computation  (eq. 2.3)
# ============================================================================

def compute_wavelet_coefficients(
    signal: np.ndarray,
    wavelet: str = WAVELET,
    level: int = J,
) -> np.ndarray:
    """
    Compute the full DWT coefficient vector for one signal.

    Applies pywt.wavedec which computes, for each scale j and translation k:

        X_ij = <X_i, psi_{j,k}>  with  psi_{j,k}(t) = 2^{j/2} psi(2^j t - k)

    and returns them concatenated as [cA_J, cD_J, cD_{J-1}, ..., cD_1],
    matching the reindexed basis {phi_{0,0}, psi_{0,0}, psi_{1,0}, ...}
    from eq. (2.3) of the paper.

    The signal is NOT normalized before projection: coefficient magnitudes
    must preserve the true energy of the signal in each basis direction,
    as required by the ranking criterion in eq. (2.4).

    Parameters
    ----------
    signal:
        Raw signal values on a uniform grid, shape (T,).
    wavelet:
        PyWavelets wavelet identifier.
    level:
        Maximum resolution level J.

    Returns
    -------
    np.ndarray
        Concatenated DWT coefficients, shape (N,) with N ≈ T.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)


# Coefficient matrix for all training signals — shape (n_train, n_coefficients)
X_coeffs = np.stack([
    compute_wavelet_coefficients(signals[i]) for i in range(len(signals))
])

print(f"Signal matrix     : {signals.shape}")
print(f"Coefficient matrix: {X_coeffs.shape}")


# ============================================================================
# Energy-based ranking  (eq. 2.4)
# ============================================================================

# Empirical energy of each coefficient j summed over the training set
energy = np.sum(X_coeffs ** 2, axis=0)

# Decreasing order: most informative basis functions first
ranking = np.argsort(energy)[::-1]

print(f"\nTop-5 coefficient indices by energy : {ranking[:5]}")
print(f"Their empirical energies            : {energy[ranking[:5]].round(4)}")


# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# One example signal (raw, not normalized)
signal_ex = signals[0]
axes[0].plot(signal_ex)
axes[0].set_title(f"Raw ECG signal  (class = {y_train[0]})")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Amplitude")

# Its full DWT coefficient vector
axes[1].stem(
    X_coeffs[0],
    markerfmt="C1o",
    linefmt="C1-",
    basefmt="k-",
)
axes[1].set_title(
    r"DWT coefficient vector  $X_i = (X_{i1}, \ldots, X_{i,2^J})$"
    f"  —  wavelet: {WAVELET}, level J={J}"
)
axes[1].set_xlabel("Coefficient index $j$")
axes[1].set_ylabel("$X_{ij}$")

# Empirical energy across all training signals, sorted by rank
axes[2].bar(range(len(energy)), energy[ranking], color="steelblue")
axes[2].set_title(
    r"Empirical energy $\sum_{i=1}^n X_{ij}^2$ sorted by rank  (eq. 2.4)"
)
axes[2].set_xlabel("Rank (0 = most energetic)")
axes[2].set_ylabel("Energy")

plt.tight_layout()
plt.show()

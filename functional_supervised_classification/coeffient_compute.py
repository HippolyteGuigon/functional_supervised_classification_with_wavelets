"""
First step toward Functional Supervised Classification with Wavelets.

Goal
----
Compute ONE wavelet coefficient exactly as described in the paper:

    <X, ψ>

where

    X  = one ECG signal
    ψ  = one wavelet

This is NOT yet the complete wavelet transform.
It is simply the projection of one ECG onto one wavelet.
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

from functional_supervised_classification.data_loading import load_ecg200 as load_data


# ============================================================================
# Load ECG200
# ============================================================================

X_train, y_train, _, _ = load_data()

# First ECG of the dataset
signal = X_train[0, 0]


# ============================================================================
# Wavelet coefficient
# ============================================================================

def wavelet_coefficient(
    signal: np.ndarray,
    wavelet: np.ndarray,
) -> float:
    """
    Compute the discrete inner product

        <X, ψ>

    Parameters
    ----------
    signal:
        ECG signal.

    wavelet:
        Wavelet sampled on the same grid.

    Returns
    -------
    float
        Wavelet coefficient.
    """

    if signal.shape != wavelet.shape:
        raise ValueError(
            "Signal and wavelet must have the same shape."
        )

    return np.dot(signal, wavelet)


# ============================================================================
# Load a Daubechies wavelet
# ============================================================================

db4 = pywt.Wavelet("db4")

# phi = scaling function
# psi = mother wavelet

phi, psi, x = db4.wavefun(level=8)


# ============================================================================
# Resample the wavelet so it has the same number of samples as the ECG
# ============================================================================

wavelet = np.interp(
    np.linspace(0, len(psi) - 1, len(signal)),
    np.arange(len(psi)),
    psi,
)


# ============================================================================
# Normalize (optional, just for visualization)
# ============================================================================

wavelet = wavelet / np.linalg.norm(wavelet)
signal_normalized = signal / np.linalg.norm(signal)


# ============================================================================
# Compute the coefficient
# ============================================================================

coefficient = wavelet_coefficient(
    signal_normalized,
    wavelet,
)

print(f"Wavelet coefficient = {coefficient:.4f}")


# ============================================================================
# Visualization
# ============================================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(10, 8),
)

axes[0].plot(signal)
axes[0].set_title(f"ECG (class = {y_train[0]})")

axes[1].plot(wavelet)
axes[1].set_title("Daubechies 4 mother wavelet")

axes[2].plot(signal_normalized, label="Normalized ECG, coefficient = {:.4f}".format(coefficient))
axes[2].plot(wavelet, label="Wavelet")
axes[2].legend()

plt.tight_layout()
plt.show()
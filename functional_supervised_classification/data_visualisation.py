import numpy as np
import matplotlib.pyplot as plt
import pywt

from functional_supervised_classification.data_loading import load_ecg200
from functional_supervised_classification.coeffient_compute import coeff_matrix, WAVELET, J

X_train, y_train, _, _ = load_ecg200()

index = 5  # Index of the sample to visualize
signal = X_train[index, 0]

signals = X_train[:, 0, :]
X_coeffs = coeff_matrix(signals)
coeffs = X_coeffs[index]

energy = np.sum(X_coeffs ** 2, axis=0)
ranking = np.argsort(energy)[::-1]

_, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(signal)
axes[0].set_title(f"Signal ECG brut  (classe = {y_train[index]})")
axes[0].set_xlabel("Pas de temps")
axes[0].set_ylabel("Amplitude")

axes[1].stem(coeffs, markerfmt="C1o", linefmt="C1-", basefmt="k-")
axes[1].set_title(
    r"Vecteur de coefficients DWT  $X_i = (X_{i1}, \ldots, X_{i,2^J})$"
    f"  —  ondelette : {WAVELET}, niveau J={J}"
)
axes[1].set_xlabel("Indice du coefficient $j$")
axes[1].set_ylabel("$X_{ij}$")

axes[2].bar(range(len(energy)), energy[ranking], color="steelblue")
axes[2].set_title(
    r"Énergie empirique $\sum_{i=1}^n X_{ij}^2$ triée par rang  (éq. 2.4)"
)
axes[2].set_xlabel("Rang (0 = plus énergétique)")
axes[2].set_ylabel("Énergie")

plt.tight_layout()
plt.savefig("ecg200_example.png", dpi=300, bbox_inches="tight")
plt.show()

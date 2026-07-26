import pywt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from functional_supervised_classification.config import load_config
from functional_supervised_classification.data_loading import load_ecg200, load_phoneme
from functional_supervised_classification.coeffient_compute import coeff_matrix, WAVELET


def plot_wavelet_cluster_view(C, y, d, dataset_name):
    """Project the best-d wavelet coefficients to 2D and scatter by class label.

    Uses PCA when d > 2 so the axes capture maximum variance, letting the eye
    judge whether natural clusters exist without running a clustering algorithm.

    Parameters
    ----------
    C            : (n, p) array of energy-ranked DWT coefficients (full matrix).
    y            : (n,) class labels.
    d            : number of leading coefficients selected by the pipeline.
    dataset_name : string used in the plot title.
    """
    X = C[:, :d]
    labels = sorted(np.unique(y))
    cmap = plt.cm.get_cmap("tab10", len(labels))

    fig, ax = plt.subplots(figsize=(7, 6))

    if d == 1:
        rng = np.random.default_rng(0)
        for i, lbl in enumerate(labels):
            mask = y == lbl
            jitter = rng.uniform(-0.25, 0.25, mask.sum())
            ax.scatter(X[mask, 0], jitter, label=str(lbl),
                       color=cmap(i), alpha=0.6, s=20)
        ax.set_xlabel("Coefficient 1 (plus énergétique)")
        ax.set_yticks([])
        proj_note = "d=1"
    elif d == 2:
        for i, lbl in enumerate(labels):
            mask = y == lbl
            ax.scatter(X[mask, 0], X[mask, 1], label=str(lbl),
                       color=cmap(i), alpha=0.6, s=20)
        ax.set_xlabel("Coefficient 1")
        ax.set_ylabel("Coefficient 2")
        proj_note = "d=2"
    else:
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X)
        var = pca.explained_variance_ratio_
        for i, lbl in enumerate(labels):
            mask = y == lbl
            ax.scatter(Z[mask, 0], Z[mask, 1], label=str(lbl),
                       color=cmap(i), alpha=0.6, s=20)
        ax.set_xlabel(f"PC 1  ({var[0]:.1%} var.)")
        ax.set_ylabel(f"PC 2  ({var[1]:.1%} var.)")
        proj_note = f"PCA 2D depuis d={d} coefficients"

    ax.set_title(
        f"Coefficients d'ondelettes sélectionnés — {dataset_name}\n"
        f"{proj_note}  |  colorié par classe"
    )
    ax.legend(title="Classe", loc="best", markerscale=1.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DATASET = load_config()["dataset"]
    assert DATASET in ["phoneme", "ecg200"], (
        f"Dataset {DATASET} not supported. Choose 'phoneme' or 'ecg200'."
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    if DATASET == "phoneme":
        X_train, y_train, _, _ = load_phoneme()
        index = 0
    else:
        X_train, y_train, _, _ = load_ecg200()
        index = 5

    # J is derived from the actual signal length and wavelet to avoid boundary effects
    J = pywt.dwt_max_level(X_train.shape[2], WAVELET)

    # ── Compute DWT coefficients ───────────────────────────────────────────────
    signal   = X_train[index, 0]
    signals  = X_train[:, 0, :]
    X_coeffs = coeff_matrix(signals, level=J)
    coeffs   = X_coeffs[index]

    energy  = np.sum(X_coeffs ** 2, axis=0)
    ranking = np.argsort(energy)[::-1]

    # ── Visualisation ──────────────────────────────────────────────────────────
    _, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(signal)
    axes[0].set_title(f"Signal brut  (classe = {y_train[index]})  —  dataset : {DATASET}")
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

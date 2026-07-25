import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from sklearn.model_selection import train_test_split


def load_ecg200():
    """
    The goal of this function is to load the ECG200 dataset from the aeon library.
    It returns the training and testing data along with their corresponding labels.

    Returns:
    X_train: np.ndarray: Training data features, shape (n, 1, 96)
    y_train: np.ndarray: Training data labels
    X_test: np.ndarray: Testing data features, shape (n, 1, 96)
    y_test: np.ndarray: Testing data labels
    """
    X_train, y_train = load_classification("ECG200", split="train")
    X_test, y_test = load_classification("ECG200", split="test")
    return X_train, y_train, X_test, y_test


def load_phoneme(train_size=250, val_size=250, random_state=42):
    """
    Load the phoneme dataset from Hastie, Buja & Tibshirani (1995), used in
    Berlinet, Biau & Rouvière (Section 3.1).

    Log-periodograms of 32 ms phoneme recordings from the TIMIT database.
    5 classes: 'aa', 'ao', 'dcl', 'iy', 'sh'. 4509 samples of length 256.

    Parameters
    ----------
    train_size:
        Number of samples in the training set (paper uses 250).
    val_size:
        Number of samples in the validation/test set (paper uses 250).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X_train: np.ndarray, shape (train_size, 1, 256)
    y_train: np.ndarray of phoneme class labels
    X_test:  np.ndarray, shape (val_size, 1, 256)
    y_test:  np.ndarray of phoneme class labels
    """
    url = "https://hastie.su.domains/ElemStatLearn/datasets/phoneme.data"
    df = pd.read_csv(url)

    feature_cols = [c for c in df.columns if c.startswith("x.")]
    X = df[feature_cols].values[:, np.newaxis, :]  # (4509, 1, 256)
    y = df["g"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=train_size,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, y_train, X_test, y_test

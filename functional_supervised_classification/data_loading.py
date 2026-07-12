from aeon.datasets import load_classification   

def load_ecg200():
    """
    The goal of this function is to load the ECG200 dataset from the aeon library. 
    It returns the training and testing data along with their corresponding labels.
    
    Returns:
    X_train: np.ndarray: Training data features
    y_train: np.ndarray: Training data labels
    X_test: np.ndarray: Testing data features
    y_test: np.ndarray: Testing data labels
    """

    X_train, y_train = load_classification("ECG200", split="train")
    X_test, y_test = load_classification("ECG200", split="test")

    return X_train, y_train, X_test, y_test
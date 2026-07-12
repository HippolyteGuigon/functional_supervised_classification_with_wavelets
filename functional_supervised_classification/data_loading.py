from aeon.datasets import load_classification   

X_train, y_train = load_classification("ECG200", split="train")
X_test, y_test = load_classification("ECG200", split="test")
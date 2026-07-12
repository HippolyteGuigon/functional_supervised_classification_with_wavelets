import matplotlib.pyplot as plt
from functional_supervised_classification.data_loading import load_ecg200

X_train, y_train, _, _ = load_ecg200()

plt.figure(figsize=(8, 4))
plt.plot(X_train[0, 0])
plt.title(f"Classe : {y_train[0]}")
plt.xlabel("Temps")
plt.ylabel("Amplitude")

# Sauvegarde en PNG
plt.savefig("ecg200_example.png", dpi=300, bbox_inches="tight")

plt.show()
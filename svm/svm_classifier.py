# svm_classifier.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from keras.datasets import mnist
import os

# Ordner für Ergebnisse anlegen
os.makedirs("results/svm", exist_ok=True)

# 1. MNIST-Daten laden (60.000 Training, 10.000 Test)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Vorverarbeitung:
#   - 28x28 Bilder in flache 784er-Vektoren umwandeln
#   - Pixelwerte normalisieren (0–255 → 0–1)
x_train = x_train.reshape((x_train.shape[0], -1)) / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)) / 255.0

# 3. Modell erstellen und trainieren (RBF-Kernel mit kleiner Gamma)
clf = svm.SVC(kernel='rbf', gamma=0.001)
clf.fit(x_train, y_train)

# 4. Vorhersage auf Testdaten
y_pred = clf.predict(x_test)

# 5. Evaluierung: Precision, Recall, F1
report = metrics.classification_report(y_test, y_pred, digits=4)
print(report)

# Bericht auch speichern
with open("results/svm/metrics_svm.txt", "w") as f:
    f.write(report)

# 6. Konfusionsmatrix erzeugen und als PNG speichern
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
plt.title("SVM – Confusion Matrix (MNIST-Subset)")
plt.savefig("results/svm/confusion_matrix_svm.png")
plt.show()

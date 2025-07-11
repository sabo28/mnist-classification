# cnn_classifier.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# Ergebnisse speichern
os.makedirs("results/cnn", exist_ok=True)

# 1. Datensatz laden
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Daten vorbereiten:
#    - neu formen (28x28x1)
#    - normalisieren
#    - Labels one-hot encodieren
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 3. Modellarchitektur definieren (klassisches, kleines CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")  # 10 Klassen (Ziffern)
])

# 4. Modell kompilieren
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# 5. Training starten
history = model.fit(
    x_train, y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=2
)

# 6. Evaluierung auf Testdaten
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# 7. Vorhersage & Metriken
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

report = classification_report(y_test, y_pred, digits=4)
print(report)

with open("results/cnn/metrics_cnn.txt", "w") as f:
    f.write(report)

# 8. Konfusionsmatrix plotten & speichern
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", ax=ax)
plt.title("CNN – Confusion Matrix (MNIST)")
plt.savefig("results/cnn/confusion_matrix_cnn.png")
plt.close()

# 9. Trainingskurven speichern
plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("results/cnn/accuracy_curve_cnn.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("results/cnn/loss_curve_cnn.png")
plt.close()

# 🧠 MNIST-Ziffernerkennung mit SVM & CNN

Dieses Projekt vergleicht zwei unterschiedliche Klassifikationsverfahren zur Erkennung handgeschriebener Ziffern auf Basis des MNIST-Datensatzes:

- **Support Vector Machine (SVM)** mit Scikit-learn
- **Convolutional Neural Network (CNN)** mit Keras & TensorFlow

Ziel ist die empirische Evaluation und der Vergleich beider Ansätze hinsichtlich ihrer Klassifikationsleistung.

---

## 📁 Projektstruktur

```
mnist-classification/
├── cnn/                      # CNN-Implementierung (Keras + TensorFlow)
│   └── cnn_classifier.py
├── svm/                      # SVM-Implementierung (Scikit-learn)
│   └── svm_classifier.py
├── results/                  # Ausgabeordner für Metriken & Visualisierungen
│   ├── cnn/
│   └── svm/
├── requirements.txt          # Python-Abhängigkeiten
├── Dockerfile                # Basisimage für beide Modelle
├── docker-compose.yml        # Separates Setup für svm / cnn
├── .gitignore
└── README.md
```

---

## 📦 Setup mit Docker Compose

### 🔧 1. Build ausführen

```bash
docker-compose build
```

### ▶️ 2. SVM-Modell ausführen

```bash
docker-compose run --rm svm
```

### ▶️ 3. CNN-Modell ausführen

```bash
docker-compose run --rm cnn
```

> Ergebnisse werden automatisch in `results/svm/` bzw. `results/cnn/` gespeichert.

---

## 🧪 Ausgaben je Modell

### Ergebnisse in `results/svm/`
- `metrics_svm.txt`: Precision, Recall, F1-Score je Klasse
- `confusion_matrix_svm.png`: Verwechslungsmatrix

### Ergebnisse in `results/cnn/`
- `metrics_cnn.txt`: Precision, Recall, F1-Score je Klasse
- `confusion_matrix_cnn.png`: Verwechslungsmatrix
- `accuracy_curve_cnn.png`: Verlauf der Genauigkeit über Epochen
- `loss_curve_cnn.png`: Verlauf des Verlusts über Epochen

---

## ⚙️ Alternativ: Lokale Ausführung (ohne Docker)

```bash
# Virtuelle Umgebung
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt

# Ausführen
python svm/svm_classifier.py
python cnn/cnn_classifier.py
```

---

## 📎 Verweis zur Hausarbeit

Dieses Projekt begleitet die Hausarbeit im Modul **DLBIKI01 – Künstliche Intelligenz** an der IU Internationale Hochschule:

> _"Analyse zweier Klassifikationsverfahren zur Mustererkennung am Beispiel des MNIST-Datensatzes"_

Die Analyse, theoretischen Grundlagen und Auswertung der Ergebnisse sind vollständig im schriftlichen Bericht dokumentiert.

---

## 📝 Lizenz

- Code: MIT-Lizenz
- MNIST-Datensatz: [CC BY-SA 3.0](http://yann.lecun.com/exdb/mnist/) (Yann LeCun et al.)

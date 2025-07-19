# 🫀 ECG Arrhythmia Detection using Deep Learning

This project builds a deep learning-based model to classify different types of arrhythmias from ECG signals. It uses the MIT-BIH Arrhythmia dataset (in CSV format) to preprocess ECG data, segment beats, and classify them using a 1D Convolutional Neural Network (CNN).

---

## 🚀 Features

- **Deep Learning Model**: Built using TensorFlow/Keras with Conv1D layers.
- **Multi-class Classification**: Supports detection of various beat types.
- **CSV Input Support**: Accepts CSV files containing ECG signals (`MLII` lead).
- **Early Stopping**: Prevents overfitting during training.
- **Visualization**: Accuracy/loss plots, class distribution pie chart, and confusion matrix.

---

## 📂 Dataset

- **Source**: [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- **Format**: CSV files (e.g., `100.csv`, `101.csv`, ...)
- **Required Columns**: 
  - `time_ms`
  - `MLII` *(used for ECG signal)*
  - `V1` *(optional)*

⚠️ Files missing the `MLII` column will be skipped automatically.

---

## 🧠 Model Architecture

```text
Input (360 samples)
    ↓
Conv1D → MaxPooling1D
    ↓
Conv1D → MaxPooling1D
    ↓
Flatten → Dense(128) → Dropout(0.5)
    ↓
Dense(softmax)
```

## 🛠️ Prerequisites
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

## ▶️ How to Run
1. Clone this repository:
``` bash
git clone https://github.com/Ghanasree-S/ECG_Arrhythmia_Detection.git
cd ECG_Arrhythmia_Detection

```
2. Ensure all CSV files are inside the dataset/ directory.
3. Run the script:
```bash
python arrhythmia.py

```

## 📊 Output
1. Training Performance: Accuracy and loss over epochs
2. Class Distribution: Pie chart of beat label frequencies
3. Confusion Matrix: Visualization of model performance
4. Classification Report: Precision, recall, F1-score per class

## 🧬 Beat Type Mapping

| Label | Description                              |
|-------|------------------------------------------|
| N     | Normal beat                              |
| L     | Left bundle branch block beat            |
| R     | Right bundle branch block beat           |
| A     | Atrial premature contraction             |
| a     | Aberrated atrial premature beat          |
| V     | Premature ventricular contraction        |
| F     | Fusion of ventricular and normal beat    |
| j     | Nodal (junctional) escape beat           |
| E     | Ventricular escape beat                  |
| /     | Paced beat                               |
| f     | Fusion of paced and normal beat          |
| Q     | Unclassified                             |
| ?     | Unknown or not classified                |


## 📝 Notes
1. Only samples with 360 time steps (centered on annotated beats) are used.
2. Labels are encoded and one-hot encoded for classification.
3. The model uses categorical crossentropy and softmax activation.



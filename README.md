# spam-detection-using-an-LSTM
# Spam Detection Using LSTM

## Overview

This project is a **Spam Detection System** that classifies SMS messages as **spam** or **ham (not spam)** using a **Long Short-Term Memory (LSTM)** deep learning model. The dataset used is the popular **SMS Spam Collection** dataset, which consists of labeled SMS messages.

The project is implemented in **TensorFlow 2** and trained using a **Bidirectional LSTM** model with **text vectorization** for preprocessing.

## Features

- Preprocessing of text messages using **TextVectorization**.
- LSTM-based neural network for SMS classification.
- Training and evaluation of the model with **accuracy metrics**.
- Performance visualization with **matplotlib**.
- Spam prediction on new messages.

---

## Dataset

The project uses the **SMS Spam Collection** dataset. If not already available, you can download it from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

The dataset contains:

- `v1`: Label (`ham` or `spam`)
- `v2`: SMS message text

---

## Installation

Ensure you have Python installed (>=3.8). To install the required dependencies, run:

```bash
pip install pandas numpy tensorflow scikit-learn matplotlib
```

---

## Usage

### Running the Jupyter Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/spam-detection-lstm.git
   cd spam-detection-lstm
   ```
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `spam_detection.ipynb` and run all cells sequentially.

### Training the Model

The model is trained using an **80-20 train-test split**. You can modify hyperparameters such as:

- `max_features = 10000` (Vocabulary size)
- `sequence_length = 100` (Max sequence length)
- `epochs = 5` (Training epochs)
- `batch_size = 32`

---

## Model Architecture

- **Embedding Layer**: Converts words into dense vectors.
- **Bidirectional LSTM**: Processes text in both forward and backward directions.
- **Dense Layer (ReLU)**: Intermediate processing.
- **Dense Layer (Sigmoid)**: Outputs probability of spam.

---

## Evaluation

After training, the model achieves high accuracy. Evaluate performance using:

```python
y_pred = (model.predict(X_test_vectorized) > 0.5).astype("int32")
print(classification_report(y_test, y_pred))
```

---

## Predicting on New Messages

You can test the model with new messages:

```python
sample_texts = ["Congratulations! You've won a free ticket to Bahamas. Call now!", 
                "Hey, are we still meeting for lunch today?"]
sample_vectorized = vectorize_layer(tf.constant(sample_texts))
predictions = model.predict(sample_vectorized)

for text, pred in zip(sample_texts, predictions):
    label = "Spam" if pred[0] > 0.5 else "Ham"
    print(f"Message: {text}\nPredicted: {label}\n")
```

---

## Results & Visualization

The training progress is visualized using **matplotlib**:

```python
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

---

## Contributing

Feel free to contribute! Fork the repository, create a branch, and submit a **Pull Request**.

---

## License

This project is open-source under the **MIT License**.

---

## Contact

For questions, reach out via [GitHub Issues](https://github.com/yourusername/spam-detection-lstm/issues).


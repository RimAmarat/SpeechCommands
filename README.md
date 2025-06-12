# 🗣️ Speech Command Recognition with CNN (M5 Architecture)

This project implements a **1D Convolutional Neural Network (CNN)** based on the **M5 architecture** to classify speech commands from the [Google Speech Commands Dataset](https://arxiv.org/abs/1804.03209). It uses `torchaudio` and `PyTorch` for loading, preprocessing, training, and evaluation.

## 📦 Dataset

We use `torchaudio.datasets.SPEECHCOMMANDS` which automatically downloads and manages the dataset.

### Subsets:
- **Training**
- **Validation**
- **Testing**

These subsets are split using the official list files provided in the dataset.

## 🧠 Model Architecture

The model is a variant of the **M5 architecture**, which uses a stack of 1D convolutions and max pooling layers to learn from raw waveforms.

### Architecture Overview:
```
Conv1d(1, 32, kernel_size=80, stride=16)
→ BatchNorm
→ ReLU
→ MaxPool1d(4)
→ Conv1d(32, 32, kernel_size=3)
→ BatchNorm
→ ReLU
→ MaxPool1d(4)
→ Conv1d(32, 64, kernel_size=3)
→ BatchNorm
→ ReLU
→ MaxPool1d(4)
→ Conv1d(64, 64, kernel_size=7, stride=16)
→ BatchNorm
→ ReLU
→ AvgPool1d(adaptive to 1)
→ Linear(64 → 35)
→ LogSoftmax
```

## 🧰 Features

- 🧠 Trains on raw waveforms (no spectrograms needed)
- 🏋️ Resampling to 8000 Hz
- 📊 Batch-normalization for training stability
- 🧮 Efficient padding and collation of variable-length audio
- 📉 Learning rate scheduler
- 🔍 Real-time waveform plotting and prediction

## 🛠️ Installation

### 🔹 Dependencies

Install the required packages:

```bash
pip install torch torchaudio matplotlib tqdm
```

### 🔹 Run the script

```bash
python speech_command_recognition.py
```

## 📊 Training & Evaluation

The training loop includes:
- Batched input waveform loading
- Resampling to 8kHz
- Forward pass through the M5 CNN
- Cross-entropy loss via `NLLLoss`
- Accuracy computation on the test set

The model trains for 2 epochs by default and logs losses and accuracy.

## 🧪 Inference

After training, you can run predictions:
```python
waveform, sr, label, *_ = test_set[i]
print(f"Expected: {label}, Predicted: {predict(waveform)}")
```

Use:
```python
ipd.Audio(waveform.numpy(), rate=sr)
```
To play the audio (in notebook environments like Jupyter/Colab).

## 🧮 Utility Functions

- `label_to_index()` – converts labels to integers
- `index_to_label()` – converts prediction index to string label
- `collate_fn()` – batch collation for `DataLoader`
- `predict(waveform)` – runs inference on single waveform

## 📈 Model Size

The model contains **~174,000 trainable parameters**, suitable for real-time or embedded applications.

## 📚 References

- ["Speech Commands Dataset"](https://arxiv.org/abs/1804.03209)
- [PyTorch torchaudio documentation](https://pytorch.org/audio/stable/)
- M5 Architecture: [arXiv:1610.00087](https://arxiv.org/abs/1610.00087)

## 📜 License

This project is provided for educational and research use only.

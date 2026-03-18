# EXPANDS-MNIST

A from-scratch PyTorch CNN trained on the [Kaggle Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) competition dataset. Includes a batch prediction script and an interactive drawing canvas for real-time inference.
To be used in concert with the Kaggle Notebook https://www.kaggle.com/code/smgrieggs/mnist-expands/

---

## Files

| File | Description |
|---|---|
| `model.py` | `DigitNet` architecture + `load_model` helper |
| `predict.py` | Batch prediction on image files from the command line |
| `draw_and_predict.py` | Interactive Tkinter canvas — draw and predict in real time |
| `requirements.txt` | Python dependencies |

The Kaggle notebook (`.ipynb`) trains the model and produces `best_digit_net.pth`.

---

## Setup

**Requirements:** Python 3.9 or newer.
```bash
# 1. Clone or download this repository
cd EXPANDS-MNIST

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU support:** if you have a CUDA-capable GPU, install the CUDA-enabled PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the default CPU wheel. The scripts automatically detect and use the GPU if available.

---

## Getting the Model Checkpoint

Train the model by running the Kaggle notebook, then download `best_digit_net.pth` using one of the following methods:

**Kaggle UI**
1. Open your notebook on kaggle.com
2. Click the **Output** tab in the right-hand panel
3. Click the **↓ Download** icon next to `best_digit_net.pth`

**Kaggle API**
```bash
pip install kaggle
kaggle kernels output <your-username>/<notebook-name> -p ./
```

Place `best_digit_net.pth` in the same directory as the scripts.

---

## Usage

### Predict on image files — `predict.py`

Accepts any image format supported by Pillow (PNG, JPEG, BMP, etc.). The image will be automatically converted to greyscale and resized to 28×28.
```bash
python predict.py digit.png
```

Predict on multiple images at once:
```bash
python predict.py one.png two.png three.png
```

**If your drawing has a white/light background** (e.g. default MS Paint canvas), pass `--invert` to flip the pixel values before inference. MNIST digits are white-on-black, so without this flag a dark-on-white drawing will look like noise to the model.
```bash
python predict.py my_seven.png --invert
```

Each image produces a side-by-side panel showing the preprocessed input and a horizontal probability bar chart. The full figure is saved as `predictions.png`.

**Full options:**
```
positional arguments:
  images               One or more image files to predict

optional arguments:
  --checkpoint PATH    Path to .pth checkpoint file (default: best_digit_net.pth)
  --invert             Invert pixel values (use for white-background drawings)
  --no-display         Skip the matplotlib window (still saves predictions.png)
```

---

### Interactive drawing canvas — `draw_and_predict.py`

Opens a black canvas. Draw a digit with your mouse — the model predicts in real time and displays a live probability bar for each class.
```bash
python draw_and_predict.py
```

**Controls:**

| Input | Action |
|---|---|
| Left-click + drag | Draw |
| Right-click | Clear canvas |
| Q or Escape | Quit |

> **Note:** the canvas uses a black background to match the MNIST convention (white digit on black). Draw your digit in white using the left mouse button.

---

## Tips for Best Results

The model was trained on MNIST, which has a specific style. Your drawings will be most accurately predicted if they follow the same conventions:

- **Fill the frame** — draw the digit large, using most of the canvas area. MNIST digits are centred and occupy most of the 28×28 field.
- **Single clean stroke** — avoid sketchy or multi-pass lines; one confident stroke per segment works best.
- **White on black** — use the drawing canvas as-is (white brush on black). If predicting from a saved file drawn on a white background, always pass `--invert`.
- **Avoid decorations** — no serifs, circles around the digit, underlines, etc.

---

## Project Structure
```
EXPANDS-MNIST/
├── model.py                 # DigitNet class definition
├── predict.py               # Batch inference script
├── draw_and_predict.py      # Interactive drawing app
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── best_digit_net.pth       # Downloaded from Kaggle (not included in repo)
```

---

## Model Architecture
```
Input (1 × 28 × 28)
  │
  ▼  ConvBlock 1 — Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool(2)
  ▼  ConvBlock 2 — Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool(2)
  ▼  Flatten → 3136-d
  ▼  Linear(3136→256) → BN → ReLU → Dropout(0.5)
  ▼  Linear(256→10) → logits
```

~200K trainable parameters. Expected accuracy on the Kaggle test set: **~99.2–99.4%**.
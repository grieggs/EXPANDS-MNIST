# predict.py
# Usage:
#   python predict.py best_digit_net.pth digit1.png digit2.png ...
#
# Accepts any image format readable by Pillow (PNG, JPEG, BMP, etc.).
# The image does NOT need to be 28x28 — it will be resized automatically.
#
# Expected image style: white or light digit on a dark/black background,
# matching the MNIST convention. If your image is inverted (dark on white,
# like drawing in MS Paint with a white canvas), pass --invert.
#
# Example:
#   python predict.py best_digit_net.pth my_seven.png --invert

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from model import load_model

#  MNIST normalisation constants (must match training) 
MEAN = 0.1307
STD  = 0.3081


def preprocess(image_path: Path, invert: bool) -> torch.Tensor:
    """
    Load an image file and convert it to a (1, 1, 28, 28) tensor
    suitable for DigitNet inference.

    Steps
    -----
    1. Open with Pillow and convert to greyscale ('L' mode = 8-bit grey)
    2. Optionally invert (for light-background drawings)
    3. Resize to 28x28 using LANCZOS (high-quality downsampling)
    4. Normalise to [0,1] then standardise with MNIST mean/std
    5. Add batch dimension: (28,28) -> (1,1,28,28)
    """
    img = Image.open(image_path).convert("L")

    if invert:
        # ImageOps.invert flips pixel values: 0->255, 255->0
        # Use this when your drawing has a white background (e.g. MS Paint default)
        img = ImageOps.invert(img)

    # LANCZOS is the highest-quality PIL resampling filter for downscaling
    img = img.resize((28, 28), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0      # [0, 1]
    arr = (arr - MEAN) / STD                            # standardise

    # unsqueeze twice: (28,28) -> (1,28,28) -> (1,1,28,28)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def predict_single(model, tensor: torch.Tensor, device: torch.device):
    """
    Run one image tensor through the model.

    Returns
    -------
    pred   : int   — predicted digit class
    probs  : ndarray shape (10,) — softmax probabilities for all classes
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)                     # (1, 10)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()  # (10,)
    return int(probs.argmax()), probs


def visualize_results(image_paths, tensors, preds, probs_list, invert):
    """Plot each input image alongside its class probability bar chart."""
    n   = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))

    # Ensure axes is always 2-D even for a single image
    if n == 1:
        axes = axes[np.newaxis, :]

    colors = ["#90CAF9"] * 10  # light blue for all bars

    for i, (path, tensor, pred, probs) in enumerate(
        zip(image_paths, tensors, preds, probs_list)
    ):
        #  Left panel: input image 
        ax_img = axes[i, 0]
        display = tensor.squeeze().numpy() * STD + MEAN
        display = np.clip(display, 0, 1)
        ax_img.imshow(display, cmap="gray", interpolation="nearest")
        ax_img.set_title(f"{path.name}\nPrediction: {pred}  ({probs[pred]:.1%})",
                         fontsize=11, fontweight="bold")
        ax_img.axis("off")

        #  Right panel: probability bar chart 
        ax_bar = axes[i, 1]
        bar_colors        = colors.copy()
        bar_colors[pred]  = "#E53935"   # highlight the predicted class in red
        bars = ax_bar.barh(range(10), probs, color=bar_colors, edgecolor="white")
        ax_bar.set_yticks(range(10))
        ax_bar.set_yticklabels([str(d) for d in range(10)])
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("Probability")
        ax_bar.set_title("Class Probabilities", fontsize=10)
        ax_bar.invert_yaxis()   # digit 0 at top
        ax_bar.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
        ax_bar.grid(axis="x", alpha=0.3)

    plt.suptitle("DigitNet Predictions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Visualization saved -> predictions.png")


def main():
    parser = argparse.ArgumentParser(description="Run DigitNet on hand-drawn digit images.")
    parser.add_argument("--checkpoint", type=str, help="Path to .pth checkpoint", default="best_digit_net.pth")
    parser.add_argument("images", type=str, nargs="+", help="Image file(s) to predict")
    parser.add_argument("--invert", action="store_true",
                        help="Invert pixel values (use for white-background drawings)")
    parser.add_argument("--no-display", action="store_true",
                        help="Skip the matplotlib window (still saves predictions.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}  (device: {device})")

    image_paths, tensors, preds, probs_list = [], [], [], []

    for img_str in args.images:
        path = Path(img_str)
        if not path.exists():
            print(f"  [skip] {path} not found.", file=sys.stderr)
            continue

        tensor       = preprocess(path, args.invert)
        pred, probs  = predict_single(model, tensor, device)
        image_paths.append(path)
        tensors.append(tensor)
        preds.append(pred)
        probs_list.append(probs)
        print(f"  {path.name:<30}  ->  {pred}   (confidence: {probs[pred]:.1%})")

    if image_paths:
        visualize_results(image_paths, tensors, preds, probs_list, args.invert)


if __name__ == "__main__":
    main()

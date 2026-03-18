# draw_and_predict.py
# An interactive drawing canvas — draw a digit with your mouse and the model
# predicts it in real time, updating the probability bars on every stroke.
#
# Usage:
#   python draw_and_predict.py best_digit_net.pth
#
# Controls:
#   Left-click + drag : draw
#   Right-click       : clear canvas
#   Q / Escape        : quit

import argparse
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from model import load_model

CANVAS_SIZE = 280          # displayed canvas in pixels (10x upscale of 28x28)
BRUSH_RADIUS = 14          # drawing brush radius in canvas pixels
MEAN, STD = 0.1307, 0.3081 # MNIST normalisation constants


class DrawApp:
    """
    Tkinter application with:
      - A square drawing canvas (white brush on black background)
      - A probability bar display updated after every stroke
      - Clear (right-click) and Quit (Q/Escape) controls
    """

    def __init__(self, root: tk.Tk, model, device: torch.device):
        self.model  = model
        self.device = device
        self.root   = root
        root.title("DigitNet — Draw a Digit")
        root.resizable(False, False)

        #  Internal PIL image (28x28) used for inference 
        # We draw on a high-res canvas for smoothness, then downsample to 28x28
        # for the model. The PIL image is the ground truth; the Tk canvas is
        # just a magnified view of it.
        self.pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

        #  Layout: drawing canvas on the left, stats panel on the right 
        frame_left  = tk.Frame(root, bg="#1a1a2e")
        frame_right = tk.Frame(root, bg="#16213e", width=260, padx=16, pady=16)
        frame_left.pack(side=tk.LEFT)
        frame_right.pack(side=tk.RIGHT, fill=tk.Y)

        # Tk canvas
        self.canvas = tk.Canvas(
            frame_left,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack()

        hint = tk.Label(frame_left,
                        text="Draw  |  Right-click: clear  |  Q: quit",
                        bg="#1a1a2e", fg="#888", font=("Helvetica", 9))
        hint.pack(pady=(4, 0))

        #  Prediction display 
        self.pred_var = tk.StringVar(value="?")
        tk.Label(frame_right, textvariable=self.pred_var,
                 font=("Helvetica", 72, "bold"),
                 bg="#16213e", fg="#E53935").pack(pady=(8, 0))

        tk.Label(frame_right, text="Prediction",
                 bg="#16213e", fg="#aaa", font=("Helvetica", 10)).pack()

        #  Probability bars (one per digit) 
        tk.Label(frame_right, text="Class probabilities",
                 bg="#16213e", fg="#aaa", font=("Helvetica", 9)).pack(pady=(16, 4))

        self.bar_vars   = []   # DoubleVar for each bar width
        self.bar_labels = []   # StringVar for each probability text

        for digit in range(10):
            row = tk.Frame(frame_right, bg="#16213e")
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=str(digit), width=2,
                     bg="#16213e", fg="white",
                     font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)

            bar_bg = tk.Frame(row, bg="#0f3460", height=16, width=180)
            bar_bg.pack(side=tk.LEFT, padx=(4, 0))
            bar_bg.pack_propagate(False)

            bar_var = tk.DoubleVar(value=0)
            bar_fill = tk.Frame(bar_bg, bg="#42A5F5", height=16)
            bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)
            self.bar_vars.append((bar_var, bar_fill))

            lbl_var = tk.StringVar(value="")
            tk.Label(row, textvariable=lbl_var,
                     bg="#16213e", fg="#aaa",
                     font=("Helvetica", 8), width=5).pack(side=tk.LEFT, padx=(4, 0))
            self.bar_labels.append(lbl_var)

        #  Mouse bindings 
        self.canvas.bind("<B1-Motion>",    self._on_drag)
        self.canvas.bind("<ButtonPress-1>", self._on_drag)   # single click counts too
        self.canvas.bind("<ButtonRelease-1>", lambda e: self._predict())
        self.canvas.bind("<Button-3>",     lambda e: self._clear())  # right-click
        root.bind("<q>",      lambda e: root.destroy())
        root.bind("<Escape>", lambda e: root.destroy())

        self.last_xy = None

    #  Drawing 

    def _on_drag(self, event):
        """Draw a filled circle at the cursor position on both the Tk canvas
        and the internal PIL image (which is what the model will see)."""
        x, y = event.x, event.y
        r = BRUSH_RADIUS

        # Tk canvas (visual feedback only)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                fill="white", outline="white")

        # PIL image (used for inference) — draw with the same radius
        self.pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        self.last_xy = (x, y)

    def _clear(self):
        """Erase everything and reset the prediction display."""
        self.canvas.delete("all")
        self.pil_img  = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
        self.pred_var.set("?")
        for lbl in self.bar_labels:
            lbl.set("")
        for _, bar_fill in self.bar_vars:
            bar_fill.place(relwidth=0)

    #  Inference 

    def _predict(self):
        """
        Downsample the PIL canvas to 28x28, run the model, and update the UI.

        The drawing is done at CANVAS_SIZE x CANVAS_SIZE for smooth interaction.
        We downsample to 28x28 using LANCZOS (same as predict.py) so the model
        sees images that look like the MNIST training examples.
        """
        small = self.pil_img.resize((28, 28), Image.LANCZOS)
        arr   = np.array(small, dtype=np.float32) / 255.0
        arr   = (arr - MEAN) / STD
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        pred = int(probs.argmax())
        self.pred_var.set(str(pred))

        # Update probability bars
        for digit, (prob, (_, bar_fill), lbl_var) in enumerate(
            zip(probs, self.bar_vars, self.bar_labels)
        ):
            bar_fill.place(relwidth=float(prob))
            bar_fill.config(bg="#E53935" if digit == pred else "#42A5F5")
            lbl_var.set(f"{prob:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Interactive digit drawing + prediction.")
    parser.add_argument("--checkpoint", type=str, help="Path to .pth checkpoint file", default="best_digit_net.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(args.checkpoint, device)
    print(f"Model loaded  (device: {device})")
    print("Draw a digit on the canvas. Right-click to clear. Q to quit.")

    root = tk.Tk()
    DrawApp(root, model, device)
    root.mainloop()


if __name__ == "__main__":
    main()

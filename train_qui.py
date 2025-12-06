# gui.py

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import queue
import time

try:
    import winsound
except:
    winsound = None

from src.train import train

IMAGE1 = "chick_1.png"
IMAGE2 = "chick_2.png"
MAX_EPOCHS = 30
BG_COLOR = "#FFE6B3"

def play_done_sound():
    if winsound:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
    else:
        print("\a")  # fallback beep


class TrainingGUI:
    def __init__(self, root):
        self.thread = None
        self.root = root
        self.root.title("Training Loop")

        self.epoch_queue = queue.Queue()
        self.current_image = 0

        self.root.configure(bg=BG_COLOR)

        # One container frame with the same background
        frame = tk.Frame(root, bg=BG_COLOR)
        frame.pack(expand=True, fill="both", padx=20, pady=20)

        # load images
        imgs = []
        for path in [IMAGE1, IMAGE2]:
            img = Image.open(path).resize((300, 300))
            imgs.append(ImageTk.PhotoImage(img))
        self.images = imgs

        # UI
        self.style = ttk.Style()
        self.style.theme_use("default")

        self.style.configure(
            "TButton",
            background="#FFD37A",
            foreground="black",
            padding=6,
            font=("Arial", 11),
        )

        self.style.map(
            "TButton",
            background=[("active", "#FFC459")]
        )

        # Image
        self.image_label = tk.Label(root, image=self.images[0], bg=BG_COLOR)
        self.image_label.pack(pady=10)

        # Epoch label
        self.epoch_var = tk.StringVar(value=f"Epoch: 0 / {MAX_EPOCHS}")
        tk.Label(
            root,
            textvariable=self.epoch_var,
            font=("Arial", 14, "bold"),
            bg=BG_COLOR
        ).pack()

        # Status label
        self.status_var = tk.StringVar(value="Status: idle")
        tk.Label(
            root,
            textvariable=self.status_var,
            font=("Arial", 12),
            bg=BG_COLOR
        ).pack(pady=5)
        # Losses label
        self.loss_var = tk.StringVar(value="")
        self.loss_label = tk.Label(
            root,
            textvariable=self.loss_var,
            font=("Arial", 12),
            bg=BG_COLOR
        )
        self.loss_label.pack_forget()
        # Start button
        self.start_button = ttk.Button(root, text="Start Training", command=self.start)
        self.start_button.pack(pady=15)

        # loopers
        self.swap_images()
        self.check_queue()

    def start(self):
        self.status_var.set("Status: training...")
        self.start_button.config(state="disabled")

        self.thread = threading.Thread(
            target=train,
            args=(self.epoch_queue,),
            daemon=True
        )
        self.thread.start()

    def swap_images(self):
        self.current_image = 1 - self.current_image
        self.image_label.config(image=self.images[self.current_image])
        self.root.after(1000, self.swap_images)

    def check_queue(self):
        try:
            while True:
                msg = self.epoch_queue.get_nowait()

                if msg["msg"] == "DONE":
                    self.status_var.set("Status: finished 🎉")
                    self.loss_label.pack_forget()
                    self.start_button.config(state="enabled")
                    play_done_sound()
                else:
                    epoch = msg["msg"]
                    train_loss = msg.get("loss", None)
                    val_loss = msg.get("val_loss", None)
                    self.epoch_var.set(f"Epoch: {epoch} / {MAX_EPOCHS}")
                    self.status_var.set("Status: training...")
                    if train_loss is not None and val_loss is not None:
                        self.loss_var.set(f"Val: {msg['val_loss']:.3f}\n Train: {msg['loss']:.3f}")
                        self.loss_label.pack(pady=5)
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)


if __name__ == "__main__":
    root = tk.Tk()
    TrainingGUI(root)
    root.mainloop()

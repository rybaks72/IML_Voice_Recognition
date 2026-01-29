# gui_inference.py

import tkinter as tk
from tkinter import ttk
import threading
import queue
import numpy as np
import torch
import sounddevice as sd
import librosa

from src.resnet_model import ResNet18
from preprocessing_pipeline.preprocessing import preprocess_data
from preprocessing_pipeline.util import fix_len, MEAN, STD

SR = 22050
CLIP_LEN = 3
THRESHOLD = 0.50
MODEL_PATH = "./models/id_453_resnet_trial_221_32_1_dropout_0.5_adamw_lr_0.0005_wd_0.01_clip_length_3_increased_aug_append.pth"

BG_COLOR = "#E8F0FF"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clip_to_spectrogram(clip: np.ndarray, sr: int, clip_len: int) -> np.ndarray:
    clip = fix_len(clip, sr * clip_len)
    S = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=128)
    S_log = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_log - MEAN) / STD
    return S_norm.astype(np.float32)


def record_audio(seconds=10, sr=SR):
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze()


@torch.no_grad()
def predict_10s(wave: np.ndarray):
    clips = preprocess_data(
        wave,
        SR,
        clip_length=CLIP_LEN,
        rms=True,
        augment=False,
        label=0
    )

    if len(clips) == 0:
        return {"error": "No voiced clips detected."}

    specs = [clip_to_spectrogram(c, SR, CLIP_LEN) for c in clips]
    x = torch.tensor(np.stack(specs)).unsqueeze(1).to(DEVICE)

    logits = net(x)
    probs = torch.softmax(logits, dim=1)[:, 1]

    p_final = probs.mean().item()
    decision = "ALLOWED" if p_final > THRESHOLD else "REJECTED"

    return {
        "n_clips": len(clips),
        "p_final": p_final,
        "decision": decision
    }


def inference_worker(q: queue.Queue):
    try:
        q.put({"status": "Recording 10 seconds..."})
        wave = record_audio(10)

        q.put({"status": "Running model..."})
        res = predict_10s(wave)

        q.put({"result": res})
    except Exception as e:
        q.put({"error": str(e)})


# ---------------- GUI ---------------- #

class InferenceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Speaker Gate")
        self.root.configure(bg=BG_COLOR)

        self.queue = queue.Queue()

        frame = tk.Frame(root, bg=BG_COLOR)
        frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.status_var = tk.StringVar(value="Status: idle")
        tk.Label(
            frame,
            textvariable=self.status_var,
            font=("Arial", 12),
            bg=BG_COLOR
        ).pack(pady=5)

        self.result_var = tk.StringVar(value="")
        tk.Label(
            frame,
            textvariable=self.result_var,
            font=("Arial", 14, "bold"),
            bg=BG_COLOR
        ).pack(pady=10)

        self.button = ttk.Button(
            frame,
            text="Record & Predict",
            command=self.start
        )
        self.button.pack(pady=15)

        self.check_queue()

    def start(self):
        self.button.config(state="disabled")
        self.result_var.set("")
        self.status_var.set("Status: starting...")
        threading.Thread(
            target=inference_worker,
            args=(self.queue,),
            daemon=True
        ).start()

    def check_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()

                if "status" in msg:
                    self.status_var.set(f"Status: {msg['status']}")

                elif "result" in msg:
                    res = msg["result"]

                    if "error" in res:
                        self.result_var.set(f"ERROR: {res['error']}")
                    else:
                        self.result_var.set(
                            f"{res['decision']}\n"
                            f"Mean prob: {res['p_final']:.3f}\n"
                            f"Clips used: {res['n_clips']}"
                        )

                    self.status_var.set("Status: idle")
                    self.button.config(state="enabled")

                elif "error" in msg:
                    self.result_var.set(f"ERROR: {msg['error']}")
                    self.button.config(state="enabled")
        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)


# -------- load model once -------- #

net = ResNet18().to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
net.load_state_dict(ckpt["net_state_dict"])
net.eval()


if __name__ == "__main__":
    root = tk.Tk()
    InferenceGUI(root)
    root.mainloop()
import torch, numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn.functional as F

"""
Basic Spectrogram Augmentations
-Input shape: (B, 1, H, W)
"""

"""
time_shift: Randomly shifts a spectrogram left or right along the time axis.
Input: x (Tensor) shape (B, 1, H, W), max_shift (int)
Output: Time-shifted spectrogram batch (Tensor, same shape)
"""
def time_shift(x, max_shift=5):
    # x: (B, 1, H, W)
    x = x.clone()
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=-1)  # shift along W (time)

"""
freq_mask: Randomly masks a frequency band in the spectrogram.
Input: x (Tensor) shape (B, 1, H, W), max_height (int)
Output: Spectrogram batch with masked frequency region (Tensor)
"""
def freq_mask(x, max_height=5):
    x = x.clone()
    B, C, H, W = x.shape
    for i in range(B):
        h = torch.randint(0, max_height + 1, (1,)).item()
        if h == 0 or h >= H:
            continue
        start = torch.randint(0, H - h + 1, (1,)).item()
        x[i, :, start:start+h, :] = 0.0
    return x

"""
time_mask: Randomly masks a time segment in the spectrogram.
Input: x (Tensor) shape (B, 1, H, W), max_width (int)
Output: Spectrogram batch with masked time region (Tensor)
"""
def time_mask(x, max_width=10):
    # x: (B, 1, H, W)
    x = x.clone()
    B, C, H, W = x.shape
    for i in range(B):
        w = torch.randint(0, max_width + 1, (1,)).item()
        if w == 0 or w >= W:
            continue
        start = torch.randint(0, W - w + 1, (1,)).item()
        x[i, :, :, start:start+w] = 0.0
    return x

"""
gauss_noise: Adds Gaussian noise to the spectrogram at a chosen SNR level.
Input: x (Tensor), snr_db (float)
Output: Noisy spectrogram batch (Tensor)
"""
def gauss_noise(x, snr_db):
    signal_pow = x.float().pow(2).mean().clamp(min=1e-12)
    snr_linear = 10 ** (snr_db / 10)

    noise_pow = signal_pow / snr_linear
    sigma = torch.sqrt(noise_pow)
    noise = torch.randn_like(x) * sigma
    return torch.clamp(x + noise, min=0.0)

"""
get_threshold_roc: Finds the best classification threshold using ROC.
Input: net (model), dataloader, device
Output: Best threshold (float), AUC score (float)
"""
def get_threshold_roc(net, dataloader, device):
    net.eval()
    all_probabilities, all_labels = [], []

    with torch.no_grad():                                       # Disable gradients for speed
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device).float().view(-1)
            logits = net(x)                                     # Raw model outputs (before sigmoid)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy() # Convert logits → probabilities → numpy
            all_probabilities.append(probs)
            all_labels.append(y.cpu().numpy())

    all_probabilities  = np.concatenate(all_probabilities)                      # Combine batches into full arrays
    all_labels = np.concatenate(all_labels)

    fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)     # Compute ROC points & matching thresholds
    auc = roc_auc_score(all_labels, all_probabilities)                  # Overall ROC quality metric = area under the roc curve = number that summarizes how well the model separates the two classes across all possible thresholds: 1 = perfect separation; 0.5 = guess game; 0.5 < = model is inverted
    #Basically if auc high =>  the model would still perform well, despite not having the exact optimal threshold
                    # if low => no threshold will save you cuz the classes overlap too much in probability space

    youden = tpr - fpr                                          # Youden's J (best balance of FPR & TPR)
    best_idx = np.argmax(youden)
    threshold = thresholds[best_idx]

    print(f"AUC={auc:.4f} | Best threshold={threshold:.4f} | TPR={tpr[best_idx]:.3f} | FPR={fpr[best_idx]:.3f}")

    return threshold, auc

"""
vtlp: Applies Vocal Tract Length Perturbation by warping frequency bins.
Input: spec (Tensor) shape (B, 1, M, T), sr (int), alpha range (float)
Output: Warped spectrogram batch (Tensor, same shape)
"""
def vtlp(spec, sr=22050, alpha_min=0.8, alpha_max=1.2):
    B, C, M, T = spec.shape
    device = spec.device

    alpha = torch.empty(B, 1, 1, 1, device=device).uniform_(alpha_min, alpha_max)

    # original mel bin positions
    orig_bins = torch.linspace(0, 1, steps=M, device=device).view(1, 1, M, 1)

    # warp curve
    f0 = 4800 / (sr/2)  # normalized cutoff
    warped = torch.where(
        orig_bins < f0,
        orig_bins * alpha,
        f0 + (orig_bins - f0) * ((1 - alpha) / (1 - f0))
    )

    warped = warped.clamp(0, 1)

    # grid for interpolation
    grid = torch.zeros(B, M, T, 2, device=device)
    grid[..., 0] = warped.squeeze(1) * 2 - 1  # x-axis (freq)
    grid[..., 1] = torch.linspace(-1, 1, steps=T, device=device).view(1, 1, T).expand(B, M, T)

    out = F.grid_sample(spec, grid, mode='bilinear', align_corners=True)
    return out
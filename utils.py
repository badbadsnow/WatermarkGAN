# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import glob
import os
import matplotlib
import torch
from torch.nn.utils import weight_norm

matplotlib.use("Agg")
import matplotlib.pylab as plt
from meldataset import MAX_WAV_VALUE
from scipy.io.wavfile import write


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_spectrogram_clipped(spectrogram, clip_max=2.0):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=1e-6,
        vmax=clip_max,
    )
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading '{filepath}'")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print(f"Saving checkpoint to {filepath}")
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix, renamed_file=None):
    # Fallback to original scanning logic first
    pattern = os.path.join(cp_dir, prefix + "????????")
    cp_list = glob.glob(pattern)

    if len(cp_list) > 0:
        last_checkpoint_path = sorted(cp_list)[-1]
        print(f"[INFO] Resuming from checkpoint: '{last_checkpoint_path}'")
        return last_checkpoint_path

    # If no pattern-based checkpoints are found, check for renamed file
    if renamed_file:
        renamed_path = os.path.join(cp_dir, renamed_file)
        if os.path.isfile(renamed_path):
            print(f"[INFO] Resuming from renamed checkpoint: '{renamed_file}'")
            return renamed_path

    return None


def save_audio(audio, path, sr):
    # wav: torch with 1d shape
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype("int16")
    write(path, sr, audio)


def calculate_metrics(true, pred):
    # 转换为Tensor
    tl = torch.tensor(true)
    pp = torch.tensor(pred)

    # 计算混淆矩阵
    pred_labels = (pp > 0.5).int()
    tp = ((pred_labels == 1) & (tl == 1)).sum().item()
    tn = ((pred_labels == 0) & (tl == 0)).sum().item()
    fp = ((pred_labels == 1) & (tl == 0)).sum().item()
    fn = ((pred_labels == 0) & (tl == 1)).sum().item()

    # 基础指标
    acc = (tp + tn) / len(tl)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # AUC计算（基于排序）
    sorted_indices = torch.argsort(pp, descending=True)
    sorted_labels = tl[sorted_indices]

    tpr_points, fpr_points = [], []
    tp_count, fp_count = 0, 0
    num_pos = tl.sum().item()
    num_neg = len(tl) - num_pos

    for i in range(len(sorted_labels)):
        if sorted_labels[i] == 1:
            tp_count += 1
        else:
            fp_count += 1
        tpr_points.append(tp_count / num_pos)
        fpr_points.append(fp_count / num_neg)

    # 梯形法积分
    auc = 0.0
    for i in range(1, len(fpr_points)):
        dx = fpr_points[i] - fpr_points[i - 1]
        y_avg = (tpr_points[i] + tpr_points[i - 1]) / 2
        auc += dx * y_avg

    return acc, tpr, fpr, auc

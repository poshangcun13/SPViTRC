import os
import pickle
import numpy as np

# ===================== 1. 路径设置 =====================
data_dir = "./deap-dataset/data_preprocessed_python"

# ===================== 2. 参数设置 =====================
fs = 128                         # 采样率
baseline_sec = 3
baseline = baseline_sec * fs     # baseline 点数 = 384
num_eeg_channels = 32

window_sec = 2                   # ★ 2 秒窗口
window_len = window_sec * fs     # 256
stride = window_len              # 非重叠窗口（顶刊最常见）

X_all = []
y_all = []

# ===================== 3. 遍历所有 .dat 文件 =====================
dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".dat")])
print(f"Found {len(dat_files)} subjects")

for dat_file in dat_files:
    dat_path = os.path.join(data_dir, dat_file)

    with open(dat_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    # 原始数据
    data = subject["data"]      # (40, 40, 8064)
    labels = subject["labels"]  # (40, 4)

    # 只取 EEG
    eeg = data[:, :num_eeg_channels, :]       # (40, 32, 8064)

    # 去 baseline（前 3 秒）
    eeg = eeg[:, :, baseline:]                 # (40, 32, 7680)

    # Valence 二分类
    valence = labels[:, 0]
    y_trial = (valence > 5).astype(np.int64)   # (40,)

    # ===================== 4. 2 s 分段 =====================
    for trial_idx in range(eeg.shape[0]):
        trial_eeg = eeg[trial_idx]             # (32, 7680)
        label = y_trial[trial_idx]

        T = trial_eeg.shape[-1]
        for start in range(0, T - window_len + 1, stride):
            segment = trial_eeg[:, start:start + window_len]  # (32, 256)
            X_all.append(segment)
            y_all.append(label)

    print(f"{dat_file} processed, segments so far: {len(X_all)}")

# ===================== 5. 合并所有被试 =====================
X_all = np.stack(X_all, axis=0)    # (N, 32, 256)
y_all = np.array(y_all)            # (N,)

print("Final EEG shape:", X_all.shape)
print("Final label shape:", y_all.shape)

# ===================== 6. 保存 =====================
np.save("processed/DEAP_EEG_2s.npy", X_all)
np.save("processed/DEAP_valence_labels_2s.npy", y_all)

print("All data saved successfully!")

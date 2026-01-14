import os
import numpy as np
import scipy.io
from tqdm import tqdm

# ===================== 1. 路径设置 =====================
data_dir = "./data IIb/data/data"  # 存放所有 .mat 文件
save_dir = "./processed"  # 保存处理后的 npy 文件
os.makedirs(save_dir, exist_ok=True)

# ===================== 2. 参数设置 =====================
fs = 240  # 采样率 240 Hz
channels = list(range(64))  # 选择所有 64 通道
window_sec = 1.0  # 1 秒窗口
window_len = int(fs * window_sec)
stride = window_len  # 非重叠窗口

X_all = []

# ===================== 3. 遍历所有被试 =====================
mat_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".mat")])
print(f"Found {len(mat_files)} subjects")

for mat_file in tqdm(mat_files):
    mat = scipy.io.loadmat(mat_file)

    # ---- 根据 Albany 数据结构提取 EEG 信号 ----
    # 假设 EEG 信号字段叫 'X' 或 'signal'
    if 'X' in mat:
        signal = mat['X']  # timepoints x channels
    elif 'signal' in mat:
        signal = mat['signal']
    else:
        raise KeyError(f"No EEG data found in {mat_file}")

    # 将所有通道选出来
    trial_data = signal[:, channels].T  # channels x timepoints

    # ---- 切分窗口 ----
    T_total = trial_data.shape[1]
    for start in range(0, T_total - window_len + 1, stride):
        segment = trial_data[:, start:start + window_len]  # channels x window_len
        X_all.append(segment)

# ===================== 4. 合并为 numpy 数组 =====================
X_all = np.array(X_all, dtype=np.float32)
X_all = X_all[:, np.newaxis, :, :]  # [samples, 1, channels, timepoints]

print("Final EEG shape:", X_all.shape)

# ===================== 5. 保存 =====================
np.save(os.path.join(save_dir, "BCI2b_P300_EEG_reconstruction.npy"), X_all)

print("All data saved successfully!")

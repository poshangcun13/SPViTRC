import os
import numpy as np
import mne

# ================== 基本配置 ==================
DATA_ROOT = "../eegmmidb"      # PhysioNet EEGMMIDB 根目录
SAVE_DIR = "./processed"      # 保存 npz 的目录
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 160          # 采样率（EEGMMIDB 固定）
N_CH = 64         # EEG 通道数
TRIAL_LEN = 512   # 每个 trial 长度（3.2s @160Hz）

# ================== 任务定义（严格官方） ==================
TASKS = {
    "MI-LR": {"runs": ["03", "07", "11"], "run_type": "H"},
    "MI-HF": {"runs": ["05", "09", "13"], "run_type": "HF"},
    "MM-LR": {"runs": ["04", "08", "12"], "run_type": "H"},
    "MM-HF": {"runs": ["06", "10", "14"], "run_type": "HF"},
}

# ================== 标签映射（永远二分类） ==================
LABEL_MAP = {
    "H": {"T1": 0, "T2": 1},  # 左拳 / 右拳
    "HF": {"T1": 0, "T2": 1},  # 双拳 / 双脚
}

# ================== 单 run 处理函数 ==================
def process_run(raw, run_type, trial_len):
    """
    从单个 Raw 中提取 trials
    返回:
        X_trials: list of (64, T)
        y_trials: list of labels (0/1)
    """
    raw.pick("eeg")
    data = raw.get_data()  # (64, T)

    events, event_id = mne.events_from_annotations(raw)

    if "T1" not in event_id or "T2" not in event_id:
        return [], []

    T1_code = event_id["T1"]
    T2_code = event_id["T2"]

    X_trials, y_trials = [], []

    for onset, _, code in events:
        if onset + trial_len > data.shape[1]:
            continue

        if code == T1_code:
            label = LABEL_MAP[run_type]["T1"]
        elif code == T2_code:
            label = LABEL_MAP[run_type]["T2"]
        else:
            continue

        trial = data[:, onset:onset + trial_len]

        # trial-level z-score
        trial = (trial - trial.mean(axis=-1, keepdims=True)) / \
                (trial.std(axis=-1, keepdims=True) + 1e-6)

        X_trials.append(trial.astype(np.float32))
        y_trials.append(label)

    return X_trials, y_trials

# ================== 主处理循环 ==================
for task_name, cfg in TASKS.items():
    print(f"\nProcessing {task_name} ...")

    X_all, y_all, s_all = [], [], []

    runs = cfg["runs"]
    run_type = cfg["run_type"]

    subjects = sorted(os.listdir(DATA_ROOT))
    for sid, subj in enumerate(subjects):
        subj_path = os.path.join(DATA_ROOT, subj)
        if not os.path.isdir(subj_path):
            continue

        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith(".edf"):
                continue

            run_id = fname[-6:-4]
            if run_id not in runs:
                continue

            raw = mne.io.read_raw_edf(
                os.path.join(subj_path, fname),
                preload=True,
                verbose=False
            )

            X_trials, y_trials = process_run(raw, run_type, TRIAL_LEN)

            X_all.extend(X_trials)
            y_all.extend(y_trials)
            s_all.extend([sid] * len(X_trials))

    if len(X_all) == 0:
        print(f"⚠ {task_name}: no trials found!")
        continue

    X_all = np.stack(X_all)
    y_all = np.array(y_all, dtype=np.int64)
    s_all = np.array(s_all, dtype=np.int64)

    save_path = os.path.join(SAVE_DIR, f"{task_name}.npz")
    np.savez(save_path, X=X_all, y=y_all, subject=s_all)

    print(f"{task_name} saved")
    print(f"  X shape : {X_all.shape}")
    print(f"  Labels  : {np.bincount(y_all)}")

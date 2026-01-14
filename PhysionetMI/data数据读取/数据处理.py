import os
import numpy as np
import mne

# ================== 配置 ==================
DATA_ROOT = "../eegmmidb"   # PhysioNet EEGMMIDB 根目录
SAVE_DIR = "./processed"     # 保存 npz 的目录
os.makedirs(SAVE_DIR, exist_ok=True)

FS = 160          # EEG 采样率
N_CH = 64

# run 分组
RUN_LR = ["03", "04", "07", "08", "11", "12"]   # 左拳 / 右拳
RUN_HF = ["05", "06", "09", "10", "13", "14"]   # 双拳 / 双脚

# trial 长度
TRIAL_LEN = {
    "MI-H": 512,
    "MI-HF": 512,
    "MM-H": 512,
    "MM-HF": 512
}

# 标签映射
LABEL_MAP = {
    "LR": { "T1": 0, "T2": 1 },   # 左拳 / 右拳
    "HF": { "T1": 2, "T2": 3 }    # 双拳 / 双脚
}

# ================== 辅助函数 ==================
def process_run(raw, run_type, trial_len):
    """
    处理一个 Raw 对象，返回 X_trials, y_trials
    run_type: "LR" 或 "HF"
    """
    raw.pick("eeg")
    data = raw.get_data()   # (64, T)

    events, event_id = mne.events_from_annotations(raw)

    # T1 / T2 对应的编码
    if "T1" not in event_id or "T2" not in event_id:
        return [], []

    T1_code = event_id["T1"]
    T2_code = event_id["T2"]

    X_trials, y_trials = [], []

    for ev in events:
        onset = ev[0]
        code = ev[2]

        if onset + trial_len > data.shape[1]:
            continue

        if run_type == "LR":
            if code == T1_code:
                label = LABEL_MAP["LR"]["T1"]
            elif code == T2_code:
                label = LABEL_MAP["LR"]["T2"]
            else:
                continue
        elif run_type == "HF":
            if code == T1_code:
                label = LABEL_MAP["HF"]["T1"]
            elif code == T2_code:
                label = LABEL_MAP["HF"]["T2"]
            else:
                continue
        else:
            continue

        trial = data[:, onset:onset + trial_len]
        # trial-level 标准化
        trial = (trial - trial.mean(axis=-1, keepdims=True)) / \
                (trial.std(axis=-1, keepdims=True) + 1e-6)
        X_trials.append(trial.astype(np.float32))
        y_trials.append(label)

    return X_trials, y_trials

# ================== 主循环 ==================
for task in ["MI-H", "MI-HF", "MM-H", "MM-HF"]:
    print(f"Processing {task} ...")
    X_all, y_all, s_all = [], [], []

    # 选择 run
    if "H" in task and task.endswith("H"):
        runs = RUN_LR
        run_type = "LR"
    else:
        runs = RUN_HF
        run_type = "HF"

    trial_len = TRIAL_LEN[task]

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

            raw = mne.io.read_raw_edf(os.path.join(subj_path, fname),
                                      preload=True, verbose=False)
            X_trials, y_trials = process_run(raw, run_type, trial_len)

            X_all.extend(X_trials)
            y_all.extend(y_trials)
            s_all.extend([sid]*len(X_trials))

    # 保存 npz
    X_all = np.stack(X_all)
    y_all = np.array(y_all, dtype=np.int64)
    subject_all = np.array(s_all, dtype=np.int64)

    save_path = os.path.join(SAVE_DIR, f"{task}.npz")
    np.savez(save_path, X=X_all, y=y_all, subject=subject_all)

    print(f"{task} saved: {X_all.shape}, labels: {dict(zip(*np.unique(y_all, return_counts=True)))}")

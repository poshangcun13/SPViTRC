import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from vit_pytorch import ViT
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import os
path = "EEG_hearing_1472.npz"
data = np.load(path, allow_pickle=True)

# 正确的 key
X = data["EEG"]             # (1472, 60, 1600)

print("Original EEG shape:", X.shape)

# ===================== 2. 截取时间维 =====================
# 取前 512 个采样点
X = X[:, :, :160]           # (1472, 60, 512)

# ===================== 3. 增加通道维 =====================
# (N,60,512) → (N,1,60,512)
X = X[:, np.newaxis, :, :]

# 转成 torch tensor
X_all = torch.tensor(X, dtype=torch.float32)

print("Loaded EEG shape:", X_all.shape)

# ===================== 2. Temporal denoise =====================
def temporal_denoise_savgol(X, window=11, polyorder=2):
    X_np = X.cpu().numpy()
    X_np = savgol_filter(X_np, window_length=window, polyorder=polyorder, axis=-1)
    return torch.tensor(X_np, dtype=X.dtype)


X_all = temporal_denoise_savgol(X_all, window=11, polyorder=2)

# ===================== 3. Normalize =====================
X_all = (X_all - X_all.mean(dim=-1, keepdim=True)) / (X_all.std(dim=-1, keepdim=True) + 1e-8)

# ===================== 4. Electrode coordinates =====================
electrode_coords = torch.tensor([[i // 8, i % 8] for i in range(64)], dtype=torch.float32)  # [64,2]


# ===================== 5. Adaptive masking =====================
def adaptive_electrode_mask(signal, mask_ratio=0.25):
    masked = signal.clone()
    batch, ch, h, w = signal.shape
    num_electrodes_to_mask = int(h * mask_ratio)
    for i in range(batch):
        electrode_var = signal[i, 0].var(dim=1)
        mask_idx = torch.argsort(electrode_var)[:num_electrodes_to_mask]
        masked[i, :, mask_idx, :] = 0
    return masked


def random_electrode_mask(signal, mask_ratio=0.25):
    masked = signal.clone()
    batch, ch, h, w = signal.shape
    num_electrodes_to_mask = int(h * mask_ratio)
    for i in range(batch):
        mask_idx = torch.randperm(h)[:num_electrodes_to_mask]
        masked[i, :, mask_idx, :] = 0
    return masked


# ===================== 6. Train/test split =====================
X_train, X_test = train_test_split(X_all, test_size=0.2, random_state=42)

# ===================== 7. Prepare DataLoaders =====================
X_train_mask = adaptive_electrode_mask(X_train)
train_dataset = TensorDataset(X_train_mask, X_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

X_test_mask_adaptive = adaptive_electrode_mask(X_test)
test_dataset_adaptive = TensorDataset(X_test_mask_adaptive, X_test)
test_loader_adaptive = DataLoader(test_dataset_adaptive, batch_size=128, shuffle=False)

X_test_mask_random = random_electrode_mask(X_test)
test_dataset_random = TensorDataset(X_test_mask_random, X_test)
test_loader_random = DataLoader(test_dataset_random, batch_size=128, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 8. ViT Reconstructor =====================
class SPVITRC(nn.Module):
    def __init__(self, width=512, coord_dim=128):
        super().__init__()
        self.width = width
        self.coord_linear = nn.Linear(2, coord_dim)
        self.vit = ViT(
            image_size=(60, width),
            patch_size=(4, 40),
            num_classes=64 * width,
            channels=1,
            dim=coord_dim,
            depth=12,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )
        self.linear_out = nn.Linear(coord_dim, width)

    def forward(self, x):
        B = x.size(0)
        coord_embed = self.coord_linear(electrode_coords.to(x.device))  # [64, dim]

        x_patches = self.vit.to_patch_embedding(x)  # [B, num_patches, dim]
        num_patches_h = 60 // 4
        num_patches_w = self.width // 40
        electrode_idx = torch.arange(num_patches_h) * 4
        coord_patch = coord_embed[electrode_idx].repeat(num_patches_w, 1)
        x_patches = x_patches + coord_patch.unsqueeze(0)

        out = self.vit.transformer(x_patches)
        out = self.vit.to_latent(out)
        out = self.linear_out(out)
        out = out.view(B, 1, 60, self.width)
        return out


model = SPVITRC(width=160, coord_dim=128).to(device)

# ===================== 9. Loss & optimizer =====================
criterion_time = nn.L1Loss()


def criterion_freq(x_out, x_target):
    X_out_f = torch.abs(torch.fft.fft(x_out, dim=-1))
    X_tar_f = torch.abs(torch.fft.fft(x_target, dim=-1))
    return nn.L1Loss()(X_out_f, X_tar_f)


optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ===================== 10. Metrics =====================
def pearson_correlation(x, y, eps=1e-8):
    batch_size = x.size(0)
    x_flat = x.view(batch_size, -1)
    y_flat = y.view(batch_size, -1)
    x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
    y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = torch.sqrt((x_centered ** 2).sum(dim=1) * (y_centered ** 2).sum(dim=1) + eps)
    corr = numerator / denominator
    return corr.mean().item()


def distance_correlation(x, y, eps=1e-8):
    n = x.size(0)
    a = torch.cdist(x.view(n, -1), x.view(n, -1), p=2)
    b = torch.cdist(y.view(n, -1), y.view(n, -1), p=2)
    A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
    B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
    dcov = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()
    return (dcov / (torch.sqrt(dvar_x * dvar_y) + eps)).item()


def evaluate(model, loader):
    model.eval()
    nmse_list, pcc_list, dc_list = [], [], []
    with torch.no_grad():
        for X_in, X_target in loader:
            X_in, X_target = X_in.to(device), X_target.to(device)
            X_out = model(X_in)
            nmse_list.append(((X_out - X_target) ** 2).sum() / (X_target ** 2).sum().clamp_min(1e-8).cpu())
            pcc_list.append(pearson_correlation(X_out, X_target))
            dc_list.append(distance_correlation(X_out, X_target))
    return torch.stack(nmse_list).mean().item(), np.mean(pcc_list), np.mean(dc_list)

# ===================== 10. Metrics (per-sample version) =====================
def evaluate_all(model, loader):
    """
    对测试集每个 batch 计算 NMSE, PCC, DC，并返回列表，便于绘制散点图
    """
    model.eval()
    nmse_list, pcc_list, dc_list = [], [], []
    with torch.no_grad():
        for X_in, X_target in loader:
            X_in, X_target = X_in.to(device), X_target.to(device)
            X_out = model(X_in)

            # NMSE
            nmse = ((X_out - X_target) ** 2).sum(dim=[1,2,3]) / (X_target ** 2).sum(dim=[1,2,3]).clamp_min(1e-8)
            nmse_list.extend(nmse.cpu().numpy())  # 保存每个样本的值

            # PCC
            batch_size = X_out.size(0)
            x_flat = X_out.view(batch_size, -1)
            y_flat = X_target.view(batch_size, -1)
            x_centered = x_flat - x_flat.mean(dim=1, keepdim=True)
            y_centered = y_flat - y_flat.mean(dim=1, keepdim=True)
            numerator = (x_centered * y_centered).sum(dim=1)
            denominator = torch.sqrt((x_centered**2).sum(dim=1) * (y_centered**2).sum(dim=1) + 1e-8)
            corr = numerator / denominator
            pcc_list.extend(corr.cpu().numpy())

            # Distance correlation
            n = X_out.size(0)
            a = torch.cdist(x_flat, x_flat, p=2)
            b = torch.cdist(y_flat, y_flat, p=2)
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dc = (A*B).mean(dim=[0,1]) / (torch.sqrt((A*A).mean(dim=[0,1])*(B*B).mean(dim=[0,1])) + 1e-8)
            dc_list.extend([dc.item()]*n)  # 对每个样本重复

    return nmse_list, pcc_list, dc_list



# ===================== 11. Training loop =====================
epochs = 10000
for epoch in range(epochs):
    model.train()
    for X_in, X_target in train_loader:
        X_in, X_target = X_in.to(device), X_target.to(device)
        optimizer.zero_grad()
        X_out = model(X_in)
        loss = criterion_time(X_out, X_target) + 0.1 * criterion_freq(X_out, X_target)
        loss.backward()
        optimizer.step()
    mse, pcc, dc = evaluate(model, test_loader_adaptive)
    print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.6f} adaptive MSE: {mse:.6f}  PCC: {pcc:.6f}  DC: {dc:.6f}")
    mse, pcc, dc = evaluate(model, test_loader_random)
    print(f"Random MSE: {mse:.6f}  PCC: {pcc:.6f}  DC: {dc:.6f}")


    # # 每100个epoch保存一次每样本指标
    # if (epoch+1)  % 200 == 0:
    #     # nmse_ad, pcc_ad, dc_ad = evaluate_all(model, test_loader_adaptive)
    #     # nmse_ra, pcc_ra, dc_ra = evaluate_all(model, test_loader_random)
    #     #
    #     # os.makedirs("评估指标序列", exist_ok=True)
    #     # np.savez(f"评估指标序列/0.75/SPViTRC_adaptive_epoch{epoch}.npz",
    #     #          nmse=np.array(nmse_ad), pcc=np.array(pcc_ad), dc=np.array(dc_ad))
    #     # np.savez(f"评估指标序列/0.75/SPViTRC_random_epoch{epoch}.npz",
    #     #          nmse=np.array(nmse_ra), pcc=np.array(pcc_ra), dc=np.array(dc_ra))
    #
    #     # ===================== 12. Save model =====================
    #     torch.save(model.state_dict(), "./地形图-0.75/SPVITRC.pth")





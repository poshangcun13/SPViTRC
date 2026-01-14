import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from vit_pytorch import ViT
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import os

# ===================== Device =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 1. Load DEAP dataset =====================
X = np.load("../processed/BCI_IV2b_EEG_2s.npy")          # (N, 32, 256)
X = X[:, np.newaxis, :, :]              # (N, 1, 32, 256)
X = torch.tensor(X, dtype=torch.float32)
# X = X.squeeze(2)


print("BCI IV EEG shape:", X.shape)

# ===================== 2. Temporal denoise =====================
def temporal_denoise_savgol(X, window=11, polyorder=2):
    X_np = X.cpu().numpy()
    X_np = savgol_filter(X_np, window_length=window,
                         polyorder=polyorder, axis=-1)
    return torch.tensor(X_np, dtype=X.dtype)

X = temporal_denoise_savgol(X)

# ===================== 3. Normalize =====================
X = (X - X.mean(dim=-1, keepdim=True)) / \
    (X.std(dim=-1, keepdim=True) + 1e-8)

electrode_coords = torch.tensor([[0, i] for i in range(3)], dtype=torch.float32)  # [3, 2]


# ===================== 5. Masking =====================
def adaptive_electrode_mask(signal, mask_ratio=0.34):
    masked = signal.clone()
    B, C, H, W = signal.shape
    num_mask = int(H * mask_ratio)

    for i in range(B):
        var = signal[i, 0].var(dim=1)
        idx = torch.argsort(var)[:num_mask]
        masked[i, :, idx, :] = 0
    return masked


def random_electrode_mask(signal, mask_ratio=0.34):
    masked = signal.clone()
    B, C, H, W = signal.shape
    num_mask = int(H * mask_ratio)

    for i in range(B):
        idx = torch.randperm(H)[:num_mask]
        masked[i, :, idx, :] = 0
    return masked

# ===================== 6. Train / Test split =====================
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# ===================== 7. DataLoaders =====================
X_train_mask = adaptive_electrode_mask(X_train)
train_loader = DataLoader(
    TensorDataset(X_train_mask, X_train),
    batch_size=256, shuffle=True, drop_last=True
)

X_test_ad = adaptive_electrode_mask(X_test)
test_loader_ad = DataLoader(
    TensorDataset(X_test_ad, X_test),
    batch_size=256, shuffle=False
)

X_test_ra = random_electrode_mask(X_test)
test_loader_ra = DataLoader(
    TensorDataset(X_test_ra, X_test),
    batch_size=256, shuffle=False
)

# ===================== 8. SPViTRC (DEAP version) =====================
class SPVITRC(nn.Module):
    def __init__(self, width=500, coord_dim=128):
        super().__init__()
        self.width = width
        self.coord_linear = nn.Linear(2, coord_dim)

        self.vit = ViT(
            image_size=(3, width),
            patch_size=(1, 100),
            num_classes=3 * width,
            channels=1,
            dim=coord_dim,
            depth=12,
            heads=6,
            mlp_dim=256,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.linear_out = nn.Linear(coord_dim, 100)

    def forward(self, x):
        B = x.size(0)

        coord_embed = self.coord_linear(
            electrode_coords.to(x.device)
        )  # [32, dim]

        x_patch = self.vit.to_patch_embedding(x)
        num_patches_h = 3 // 1
        num_patches_w = self.width // 100
        electrode_idx = torch.arange(num_patches_h) * 1
        coord_patch = coord_embed[electrode_idx].repeat(num_patches_w, 1)
        x_patches = x_patch + coord_patch.unsqueeze(0)

        out = self.vit.transformer(x_patches)
        out = self.vit.to_latent(out)
        out = self.linear_out(out)
        out = out.view(B, 1, 3, self.width)
        return out


model = SPVITRC(width=500,coord_dim=128).to(device)

# ===================== 9. Loss & optimizer =====================
criterion_time = nn.L1Loss()

def criterion_freq(x, y):
    xf = torch.abs(torch.fft.fft(x, dim=-1))
    yf = torch.abs(torch.fft.fft(y, dim=-1))
    return nn.L1Loss()(xf, yf)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

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

# ===================== 11. Training =====================
epochs = 500
for ep in range(epochs):
    model.train()
    for xin, xt in train_loader:
        xin, xt = xin.to(device), xt.to(device)
        optimizer.zero_grad()
        xo = model(xin)
        loss = criterion_time(xo, xt) + 0.1 * criterion_freq(xo, xt)
        loss.backward()
        optimizer.step()

    mse, pcc,dc = evaluate(model, test_loader_ad)
    print(f"Epoch [{ep + 1}/{epochs}] Loss: {loss.item():.6f} adaptive MSE: {mse:.6f}  PCC: {pcc:.6f}  DC: {dc:.6f}")

    mse, pcc,dc = evaluate(model, test_loader_ra)
    print(f"           Random   NMSE: {mse:.6f}, PCC: {pcc:.4f}, DC: {dc:.4f}")

    # # 每100个epoch保存一次每样本指标
    # if (ep + 1) % 300 == 0:
    #     nmse_ad, pcc_ad, dc_ad = evaluate_all(model, test_loader_ad)
    #     nmse_ra, pcc_ra, dc_ra = evaluate_all(model, test_loader_ra)
    #
    #     os.makedirs("评估指标序列", exist_ok=True)
    #     np.savez(f"评估指标序列/0.25/SPViTRC_adaptive_epoch{ep}.npz",
    #              nmse=np.array(nmse_ad), pcc=np.array(pcc_ad), dc=np.array(dc_ad))
    #     np.savez(f"评估指标序列/0.25/SPViTRC_random_epoch{ep}.npz",
    #              nmse=np.array(nmse_ra), pcc=np.array(pcc_ra), dc=np.array(dc_ra))
    #
    #     # ===================== 12. Save model =====================
    #     torch.save(model.state_dict(), "./地形图-0.25/SPVITRC.pth")

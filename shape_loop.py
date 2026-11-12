import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _logit(x, eps=1e-6):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x) - torch.log(1 - x)

def _atanh(x, eps=1e-6):
    x = x.clamp(-1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

class PolarFourierRenderer(nn.Module):
    """
    z = [p, cx, cy, r0, a1..aK, b1..bK]  ->  mask in (0,1)
    坐标系：grid_x, grid_y in [-1,1]
    """
    def __init__(self, H: int, W: int, K: int = 8, sharpness: float = 40.0, r_max: float = 1.25):
        super().__init__()
        self.H, self.W, self.K = H, W, K
        self.sharpness = sharpness
        self.r_max = r_max

        ys = torch.linspace(-1.0, 1.0, H)
        xs = torch.linspace(-1.0, 1.0, W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", gx[None, None, ...])  # [1,1,H,W]
        self.register_buffer("grid_y", gy[None, None, ...])  # [1,1,H,W]
        self.register_buffer("ks", torch.arange(1, K + 1).view(1, K, 1, 1))  # [1,K,1,1]

    def decode(self, z):
        B = z.shape[0]
        p_raw  = z[:, 0:1]
        cx_raw = z[:, 1:2]
        cy_raw = z[:, 2:3]
        r_raw  = z[:, 3:4]
        ab     = z[:, 4:]  # [B,2K]

        p  = torch.sigmoid(p_raw)              # [0,1]
        cx = torch.tanh(cx_raw)                # [-1,1]
        cy = torch.tanh(cy_raw)                # [-1,1]
        r0 = self.r_max * torch.sigmoid(r_raw) # [0,r_max]

        a = ab[:, :self.K]
        b = ab[:, self.K:]
        # 系数幅度别太大（否则边界会自交/抖），先用 0.15*r_max 约束
        # a = 0.15 * self.r_max * torch.tanh(a)
        # b = 0.15 * self.r_max * torch.tanh(b)
        k = torch.arange(1, self.K + 1, device=z.device, dtype=z.dtype).view(1, self.K)  # 1..K
        w = (1.0 / k)  # 高频更小，低频更大

        base = 0.25 * self.r_max  # 低频最大幅度
        a = base * w * torch.tanh(a)
        b = base * w * torch.tanh(b)
        return p, cx, cy, r0, a, b

    def forward(self, z):
        B = z.shape[0]
        p, cx, cy, r0, a, b = self.decode(z)

        dx = self.grid_x - cx.view(B, 1, 1, 1)
        dy = self.grid_y - cy.view(B, 1, 1, 1)
        rho = torch.sqrt(dx * dx + dy * dy + 1e-8)         # [B,1,H,W]
        theta = torch.atan2(dy, dx)                         # [B,1,H,W]

        theta_k = self.ks.to(theta.device) * theta          # [B,K,H,W]
        cos_k = torch.cos(theta_k)
        sin_k = torch.sin(theta_k)

        a = a.view(B, self.K, 1, 1)
        b = b.view(B, self.K, 1, 1)

        r = r0.view(B, 1, 1, 1) + (a * cos_k + b * sin_k).sum(dim=1, keepdim=True)
        r = torch.clamp(r, min=1e-3, max=self.r_max)        # 稳定一点

        mask = torch.sigmoid(self.sharpness * (r - rho))    # inside: rho<r => ~1
        mask = p.view(B, 1, 1, 1) * mask                    # 允许“无病灶”
        return mask

class MaskShapeEncoder(nn.Module):
    def __init__(self, z_dim: int, r_max: float = 1.25):
        super().__init__()
        self.r_max = r_max
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(inplace=True),
            nn.Linear(128, z_dim),
        )

    def forward(self, m):
        z = self.net(m)  # 先让网络输出一版（主要用于 ab 系数）

        B, _, H, W = m.shape
        mm = m.clamp(0.0, 1.0)

        # --- 软质心 (cx,cy) in [-1,1] ---
        ys = torch.linspace(-1.0, 1.0, H, device=m.device, dtype=m.dtype)
        xs = torch.linspace(-1.0, 1.0, W, device=m.device, dtype=m.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]
        gx = gx[None, None, ...].expand(B, 1, H, W)
        gy = gy[None, None, ...].expand(B, 1, H, W)

        mass = mm.sum(dim=(2, 3), keepdim=True)  # [B,1,1,1]
        mass_safe = mass + 1e-6
        cx = (mm * gx).sum(dim=(2, 3), keepdim=True) / mass_safe
        cy = (mm * gy).sum(dim=(2, 3), keepdim=True) / mass_safe

        valid = (mass > 5.0).to(m.dtype)  # 面积太小当作空
        cx = (cx * valid).view(B)
        cy = (cy * valid).view(B)

        # --- 用面积估 r0（等效圆半径，转到归一化坐标）---
        area_pix = mass.view(B)  # 像素面积（软）
        r_pix = torch.sqrt(area_pix / math.pi)              # 像素半径
        r0 = r_pix * (2.0 / max(1, (H - 1)))                # 转到 [-1,1] 坐标系下的半径
        r0 = r0.clamp(1e-3, self.r_max - 1e-4)

        # --- p：是否存在前景 ---
        p = valid.view(B).clamp(0, 1) * 0.98 + 0.01         # 避免 logit 变 inf

        # 写回到 z 的 raw 空间（对应 renderer.decode 的 sigmoid/tanh）
        z[:, 0] = _logit(p)                     # p_raw
        z[:, 1] = _atanh(cx)                    # cx_raw
        z[:, 2] = _atanh(cy)                    # cy_raw
        z[:, 3] = _logit(r0 / self.r_max)       # r_raw

        return z

class SPLHead(nn.Module):
    """从 backbone 最深层 feature -> z"""
    def __init__(self, z_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, z_dim)

    def forward(self, feat_last):
        x = self.pool(feat_last).flatten(1)
        x = F.relu(self.fc1(x), inplace=True)
        z = self.fc2(x)
        return z

def soft_dice_loss(prob, target, eps=1e-6):
    # prob/target: [B,1,H,W]
    inter = (prob * target).sum(dim=(1,2,3)) * 2.0
    denom = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return 1.0 - (inter / denom).mean()

class ShapeClosedLoop(nn.Module):
    """
    输入：feats(list)，用 feats[0] 回归 z1
    输出：z1, m1, z2, m2
    """
    def __init__(self, out_hw: int = 224, K: int = 8, sharpness: float = 40.0):
        super().__init__()
        self.K = K
        self.z_dim = 4 + 2*K
        self.spl = SPLHead(self.z_dim)
        self.renderer = PolarFourierRenderer(out_hw, out_hw, K=K, sharpness=sharpness)
        # self.mask_encoder = MaskShapeEncoder(self.z_dim)
        self.mask_encoder = MaskShapeEncoder(self.z_dim, r_max=self.renderer.r_max)

    def forward(self, feats):
        feat_last = feats[0]
        z1 = self.spl(feat_last)
        m1 = self.renderer(z1)

        # z2 = self.mask_encoder(m1)
        # m2 = self.renderer(z2)
        m1_det = m1.detach()
        # 可选：让面积/质心更“硬”，避免 r0 被 soft mask 低估导致小圆点
        m1_det = (m1_det > 0.5).float()

        z2 = self.mask_encoder(m1_det)
        m2 = self.renderer(z2)

        return z1, m1, z2, m2

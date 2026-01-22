import torch
import torch.nn.functional as F

_projection_layer = None  # 全局静态变量（避免每次都创建）

def cosine_alignment_loss(prompt, text_embed, labels):
    global _projection_layer

    device = prompt.device
    B, prompt_dim = prompt.shape
    _, text_dim = text_embed.shape

    # 初始化一次性线性层
    if _projection_layer is None or _projection_layer.weight.shape[1] != text_dim:
        _projection_layer = torch.nn.Linear(text_dim, prompt_dim).to(device)

    normalized_prompt = F.normalize(prompt, dim=-1)
    target_text = F.normalize(_projection_layer(text_embed[labels]), dim=-1)

    cosine_sim = (normalized_prompt * target_text).sum(dim=-1)  # [B]
    return 1.0 - cosine_sim.mean()


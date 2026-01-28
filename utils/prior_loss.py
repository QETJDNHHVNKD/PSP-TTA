import torch
import torch.nn.functional as F

def dice_loss_per_channel(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.size(0), -1)
    target = target.contiguous().view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def focal_loss_per_channel(pred, target, alpha=0.25, gamma=2.0, eps=1e-6):
    pred = pred.clamp(min=eps, max=1 - eps)
    pt = torch.where(target == 1, pred, 1 - pred)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean(dim=[1, 2])

class SelectedLoss(torch.nn.Module):
    def __init__(self, mode='dice', alpha=0.25, gamma=2.0):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor, labelseq: torch.Tensor):
    
        B = pred.shape[0]
        total_loss = 0.0
        for b in range(B):
            cls_ids = labelseq[b]
            if cls_ids.ndim == 0:
                cls_ids = cls_ids.unsqueeze(0)
            loss_sum = 0.0
            for cls_id in cls_ids:
                pred_c = pred[b, cls_id]
                targ_c = target[b, cls_id]
                if self.mode == 'dice':
                    loss_sum += dice_loss_per_channel(pred_c, targ_c)
                elif self.mode == 'bce':
                    loss_sum += F.binary_cross_entropy_with_logits(pred_c, targ_c)
                elif self.mode == 'focal':
                    pred_sigmoid = torch.sigmoid(pred_c)
                    loss_sum += focal_loss_per_channel(pred_sigmoid, targ_c, self.alpha, self.gamma)
            total_loss += loss_sum / len(cls_ids)
        return total_loss / B

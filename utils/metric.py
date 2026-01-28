
import numpy as np
import torch
import torch.nn.functional as F

def softmax_and_threshold(logits):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    return preds, probs

def compute_dice(pred, target, num_classes):
    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = 1.0 if union.item() == 0 else (2 * intersection + 1e-6) / (union + 1e-6)
        dice_per_class.append(dice.item() if isinstance(dice, torch.Tensor) else dice)  

    return dice_per_class

def compute_iou(pred, target, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = 1.0 if union.item() == 0 else intersection / (union + 1e-6)
        iou_per_class.append(iou.item() if isinstance(iou, torch.Tensor) else iou)
    return iou_per_class

def compute_pixel_accuracy(pred, target):
    correct = (pred == target).float()
    return correct.sum() / torch.numel(correct)



def evaluate_metrics(logits, labels, eps: float = 1e-6,num_classes=None):
    with torch.no_grad():
        B, C, H, W = logits.shape

        if C == 1:
            prob_fg = torch.sigmoid(logits).squeeze(1)          
        else:
            prob = torch.softmax(logits, dim=1)                 
            if prob.shape[1] < 2:
                raise ValueError("For C>=2, expect at least 2 channels with class-1 as foreground.")
            prob_fg = prob[:, 1, :, :]                 

        pred_fg = (prob_fg > 0.5).float()                       
        true_fg = (labels > 0).float()                           
        if true_fg.shape[-2:] != pred_fg.shape[-2:]:
            true_fg = F.interpolate(true_fg.unsqueeze(1), size=pred_fg.shape[-2:], mode="nearest").squeeze(1)

        valid = (true_fg.sum(dim=(1, 2)) > 0)
        if valid.sum() == 0:
            return {
                "dice_per_class": [0.0],  
                "dice_mean": 0.0,
                "iou_per_class": [0.0],
                "iou_mean": 0.0,
                "pixel_acc": float((pred_fg == true_fg).float().mean().item()),
            }

        pred_v = pred_fg[valid]
        true_v = true_fg[valid]

       
        inter = (pred_v * true_v).sum(dim=(1, 2))
        pred_sum = pred_v.sum(dim=(1, 2))
        true_sum = true_v.sum(dim=(1, 2))

        dice = ((2.0 * inter + eps) / (pred_sum + true_sum + eps))        
        union = pred_sum + true_sum - inter
        iou  = ((inter + eps) / (union + eps))                              

    
        pixel_acc = (pred_v == true_v).float().mean().item()

        dice_mean = dice.mean().item()
        iou_mean  = iou.mean().item()
        return {
            "dice_per_class": [dice_mean],
            "dice_mean": dice_mean,
            "iou_per_class": [iou_mean],
            "iou_mean": iou_mean,
            "pixel_acc": pixel_acc,
        }

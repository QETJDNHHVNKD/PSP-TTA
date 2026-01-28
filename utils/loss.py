import torch
import numpy as np

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
        dice_per_class.append(dice.item())
    return dice_per_class

def compute_iou(pred, target, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        iou = 1.0 if union.item() == 0 else intersection / (union + 1e-6)
        iou_per_class.append(iou.item())
    return iou_per_class

def compute_pixel_accuracy(pred, target):
    correct = (pred == target).float()
    return correct.sum() / torch.numel(correct)

def evaluate_metrics(logits, labels, num_classes):
    preds, _ = softmax_and_threshold(logits)
    dice_scores = compute_dice(preds, labels, num_classes)
    iou_scores = compute_iou(preds, labels, num_classes)
    pixel_acc = compute_pixel_accuracy(preds, labels).item()
    return {
        "dice_per_class": dice_scores,
        "dice_mean": np.mean(dice_scores),
        "iou_per_class": iou_scores,
        "iou_mean": np.mean(iou_scores),
        "pixel_acc": pixel_acc,
    }


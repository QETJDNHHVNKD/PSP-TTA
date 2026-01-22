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



# import torch
# import torch.nn.functional as F
# import torch.nn as nn


# class SelectedDSCLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(SelectedDSCLoss, self).__init__()
#         self.smooth = smooth
#
#     def forward(self, predict, target, labelseq):
#         """
#         predict: [B, 1, H, W]
#         target:  [B, 1, H, W] or larger
#         labelseq: [B]
#         """
#
#         assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
#
#         # 尺寸对齐
#         if predict.shape[-2:] != target.shape[-2:]:
#             target = F.interpolate(target.float(), size=predict.shape[-2:], mode='nearest')
#
#         predict = torch.sigmoid(predict)
#
#         dsc_loss_list = []
#         B = predict.shape[0]
#
#         for b in range(B):
#             pred = predict[b, 0, :, :].contiguous().view(-1)
#             targ = target[b, 0, :, :].contiguous().view(-1)
#
#             num = torch.sum(pred * targ)
#             den = torch.sum(pred) + torch.sum(targ) + self.smooth
#
#             dice_score = 2 * num / den
#             dsc_loss_list.append(1 - dice_score)
#
#         return torch.stack(dsc_loss_list).mean()
#
#
# # class SelectedBCELoss(nn.Module):
# #     def __init__(self):
# #         super(SelectedBCELoss, self).__init__()
# #         self.criterion = nn.BCEWithLogitsLoss()
# #
# #     def forward(self, predict, target, labelseq):
# #
# #         bce_loss_list = []
# #         B = len(labelseq)
# #         for b in range(B):
# #             bce_loss_list.append(self.criterion(predict[b, labelseq[b].tolist(), :], target[b, labelseq[b].tolist(),:]))
# #
# #         bce_loss_sum = torch.stack(bce_loss_list).sum()
# #         return bce_loss_sum
#
# class SelectedFLoss(nn.Module):                  #类别不平衡问题，使模型更关注难分类样本
#     def __init__(self, alpha = 0.25, gamma = 2):                      #alpha用于平衡正负样本权重的参数，gamma调节难易样本权重的聚焦参数
#         super(SelectedFLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, predict, target, labelseq):
#         probab = predict.sigmoid()
#         f_loss_list = []
#
#
#         # B = len(labelseq)
#
#         for b in range(predict.shape[0]):
#             pred = predict[b, 0, :, :]
#             targ = target[b, 0, :, :]
#             prob = probab[b, 0, :, :]
#
#             f_loss = F.binary_cross_entropy_with_logits(pred, targ, reduction="none")
#             p_t = prob * targ + (1 - prob) * (1 - targ)
#             loss = f_loss * ((1 - p_t) ** self.gamma)
#
#             alpha_t = self.alpha * targ + (1 - self.alpha) * (1 - targ)
#             loss = alpha_t * loss
#             f_loss_list.append(loss.mean())
#
#         return torch.stack(f_loss_list).sum() / len(f_loss_list)
#
#
# class SelectedCLoss(nn.Module):
#     def __init__(self, margin=1.0, alpha=0.5, beta=1.0):
#         super(SelectedCLoss, self).__init__()
#         self.margin = margin
#         self.alpha = alpha
#         self.beta = beta
#
#     def forward(self, embeddings1, embeddings2):
#         euclidean_distance = F.pairwise_distance(embeddings1, embeddings2, p=2)
#
#         boundary_samples = torch.where(euclidean_distance < self.margin, torch.ones_like(euclidean_distance),torch.zeros_like(euclidean_distance))
#         boundary_weights = self.alpha * boundary_samples + self.beta * (1 - boundary_samples)
#
#         positive_pairs = torch.exp(-boundary_weights * (euclidean_distance - self.margin))
#         negative_pairs = torch.exp(boundary_weights * (euclidean_distance + self.margin))
#
#         loss_contrastive = torch.mean(torch.log(1 + positive_pairs * negative_pairs))
#
#         return loss_contrastive
#
#
#

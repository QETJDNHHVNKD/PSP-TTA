import torch
import torch.nn.functional as F

def weighted_dice_loss(inputs, targets, class_weights):
    smooth = 1e-5
    num_classes = inputs.size(1)
    loss = 0.0

    inputs = torch.sigmoid(inputs)

    for c in range(num_classes):
        input_flat = inputs[:, c].contiguous().view(-1)
        target_flat = targets[:, c].contiguous().view(-1)
        intersection = (input_flat * target_flat).sum()
        denom = input_flat.sum() + target_flat.sum()
        dice = (2. * intersection + smooth) / (denom + smooth)
        loss += class_weights[c] * (1 - dice)

    return loss / class_weights.sum()

from torch.nn.functional import *
import torch.nn as nn
import torch

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        p = logits.view(-1, 1)
        t = targets.view(-1, 1)
        loss1 = torch.nn.functional.binary_cross_entropy_with_logits(p, t, reduction='mean')

        label = targets.long()
        # label2 = label.float()
        mask = label.float()
        num_positive = torch.sum((mask == 1).float()).float()
        num_negative = torch.sum((mask == 0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        prediction = sigmoid(logits)
        cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduce=False)

        # DICE
        num = targets.size(0)
        smooth = 1

        probs = sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num

        return score + torch.sum(cost) / (num_positive + num_negative) #（0.2、0.8）#只能在这里调整参数
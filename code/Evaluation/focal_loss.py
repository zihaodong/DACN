import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=2, alpha=None, reduction='elementwise_mean'):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = _input#torch.sigmoid(_input)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
            pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            loss = loss * self.alpha
        if self.reduction == 'elementwise_mean':
            loss = loss
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


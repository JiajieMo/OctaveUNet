"""Focal loss with logits."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
from torch import nn
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


def focal_loss_with_logits(inputs, targets, alpha=0.5, gamma=2.0,
                           reduction='mean'):
    """Focal loss with logits."""
    weight_map = alpha * targets + (1 - targets) * (1 - alpha)

    bce_loss = F.binary_cross_entropy_with_logits(input=inputs, target=targets,
                                                  reduction='none')
    focal_loss = weight_map * torch.pow((1 - torch.exp(-bce_loss)),
                                        gamma) * bce_loss

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        focal_loss = torch.mean(focal_loss)
    elif reduction == 'sum':
        focal_loss = torch.sum(focal_loss)
    else:
        LOGGER.error('Invalid reduction mode: %s', reduction)
        raise NotImplementedError(
            'Invalid reduction mode: {}'.format(reduction))

    return focal_loss


class FocalWithLogitsLoss(nn.Module):
    """Focal loss with logits."""

    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    # pylint: disable=arguments-differ
    def forward(self, logits, targets):
        loss = focal_loss_with_logits(inputs=logits, targets=targets,
                                      alpha=self.alpha, gamma=self.gamma,
                                      reduction=self.reduction)
        return loss

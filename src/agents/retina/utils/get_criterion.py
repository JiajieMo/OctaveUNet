"""Get loss criterion related objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch

from src.configs.config_node import ConfigNode
from src.agents.retina.utils.get_data import get_pil_datasets
from src.datasets.utils.get_dataset_statistics import get_positive_weight
from src.criterions.focal_loss import FocalWithLogitsLoss


LOGGER = logging.getLogger(__name__)


def get_criterion(configs: ConfigNode):
    """Get criterion."""
    criterion_name = configs.LOSS.LOSS_NAME

    if criterion_name == 'binary_cross_entropy':
        criterion = torch.nn.BCEWithLogitsLoss(
            weight=None, reduction='mean', pos_weight=None)

    elif criterion_name == 'weighted_binary_cross_entropy':
        pos_weight_factor = configs.LOSS.POS_WEIGHT_FACTOR
        target_key = configs.DATA.DATASET.TARGET_KEY
        dataset = get_pil_datasets(configs)
        pos_weight = pos_weight_factor * get_positive_weight(
            dataset=dataset, target_key=target_key)

        LOGGER.debug('Set positive weight dampening factor: %s',
                     pos_weight_factor)
        LOGGER.info('Set final positive weight: %s', pos_weight)

        criterion = torch.nn.BCEWithLogitsLoss(
            weight=None, reduction='mean', pos_weight=pos_weight)

    elif criterion_name == 'focal':
        alpha = configs.LOSS.FOCAL_ALPHA
        gamma = configs.LOSS.FOCAL_GAMMA
        criterion = FocalWithLogitsLoss(
            alpha=alpha, gamma=gamma, reduction='mean')

        LOGGER.info('Set focal alpha: %s', alpha)
        LOGGER.info('Set focal gamma: %s', gamma)

    else:
        LOGGER.error('Invalid criterion_name: %s', criterion_name)
        raise NotImplementedError

    LOGGER.info('Retrieved loss: %s', criterion_name)
    return criterion

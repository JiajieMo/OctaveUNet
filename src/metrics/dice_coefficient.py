"""Dice coefficient."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from src.datasets.utils.check_data_integrity import check_probability_map
from src.datasets.utils.check_data_integrity import check_binary_map
from src.metrics.utils.convert_data import convert_to_ndarray

LOGGER = logging.getLogger(__name__)


def get_dice_coefficient(prob_maps, binary_maps, targets, masks=None,
                         reduction='mean', epsilon=1e-8):
    """Dice coefficient."""

    assert prob_maps.shape == targets.shape
    assert binary_maps.shape == targets.shape
    assert check_probability_map(prob_maps)
    assert check_binary_map(binary_maps)
    assert check_binary_map(targets)

    batch_size = prob_maps.shape[0]
    prob_maps = prob_maps.view(batch_size, -1)
    binary_maps = binary_maps.view(batch_size, -1)
    targets = targets.view(batch_size, -1)

    prob_maps = convert_to_ndarray(prob_maps)
    binary_maps = convert_to_ndarray(binary_maps)
    targets = convert_to_ndarray(targets)

    if masks is not None:
        masks = masks.view(batch_size, -1)
        assert binary_maps.shape == masks.shape
        assert check_binary_map(masks)
        masks = convert_to_ndarray(masks)
        prob_maps = prob_maps * masks
        binary_maps = binary_maps * masks
        targets = targets * masks

    intesection = (binary_maps * targets).sum(-1)
    binary_maps_norm = binary_maps.sum(-1)
    targets_norm = binary_maps.sum(-1)

    dice_score = (2.0 * intesection + epsilon) / (
        binary_maps_norm + targets_norm + epsilon)

    if reduction == 'none':
        pass

    elif reduction == 'sum':
        dice_score = dice_score.sum()

    elif reduction == 'mean':
        dice_score = dice_score.mean()

    else:
        LOGGER.error('Invalid reduction mode: %s', reduction)
        raise NotImplementedError('Invalid reduction mode: {}'.format(
            reduction))

    return dice_score

"""Binary confusion matrix and various metrics based on it."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

from src.datasets.utils.check_data_integrity import check_binary_map
from src.metrics.utils.convert_data import convert_to_ndarray

LOGGER = logging.getLogger(__name__)


def get_binary_confusion_matrix(binary_maps, targets, masks=None,
                                reduction='sum'):
    """Get binary confusion matrix (TP, FP, TN, FN)."""
    binary_maps = convert_to_ndarray(binary_maps)
    targets = convert_to_ndarray(targets)

    assert binary_maps.shape == targets.shape
    assert check_binary_map(binary_maps)
    assert check_binary_map(targets)

    if masks is not None:
        masks = convert_to_ndarray(masks)
        assert binary_maps.shape == masks.shape
        assert check_binary_map(masks)

    targets_neg = -1 * (targets - 1)
    inputs_neg = -1 * (binary_maps - 1)

    true_pos = targets * binary_maps
    false_pos = targets_neg * binary_maps

    true_neg = targets_neg * inputs_neg
    false_neg = targets * inputs_neg

    if masks is not None:
        true_pos = true_pos * masks
        false_pos = false_pos * masks
        true_neg = true_neg * masks
        false_neg = false_neg * masks

    if reduction == 'none':
        pass

    elif reduction == 'sum':
        true_pos = float(np.sum(true_pos))
        false_pos = float(np.sum(false_pos))
        true_neg = float(np.sum(true_neg))
        false_neg = float(np.sum(false_neg))

    elif reduction == 'mean':
        true_pos = float(np.mean(true_pos))
        false_pos = float(np.mean(false_pos))
        true_neg = float(np.mean(true_neg))
        false_neg = float(np.mean(false_neg))

    else:
        LOGGER.error('Invalid reduction mode: %s', reduction)
        raise NotImplementedError('Invalid reduction mode: {}'.format(
            reduction))

    return true_pos, false_pos, true_neg, false_neg


def get_accuracy(true_pos, false_pos, true_neg, false_neg, epsilon=1e-8):
    """Get accuracy from confusion matrix."""
    denominator = true_pos + false_pos + true_neg + false_neg
    if denominator == 0:
        acc = (true_pos + true_neg) / (denominator + epsilon)
    else:
        acc = (true_pos + true_neg) / denominator
    return acc


def get_true_positive_rate(true_pos, false_neg, epsilon=1e-8):
    """Get true positive rate (sensitivity, recall)."""
    denominator = true_pos + false_neg
    if denominator == 0:
        tpr = true_pos / (denominator + epsilon)

    else:
        tpr = true_pos / denominator

    return tpr


def get_true_negative_rate(true_neg, false_pos, epsilon=1e-8):
    """Get true negative rate (specificity)."""
    denominator = true_neg + false_pos
    if denominator == 0:
        tnr = true_neg / (denominator + epsilon)
    else:
        tnr = true_neg / denominator

    return tnr


def get_precision(true_pos, false_pos, epsilon=1e-8):
    """Get precision."""
    denominator = true_pos + false_pos
    if denominator == 0:
        prc = true_pos / (denominator + epsilon)
    else:
        prc = true_pos / denominator

    return prc


def get_prevalence(true_pos, false_pos, true_neg, false_neg, epsilon=1e-8):
    """Get prevalence."""
    denominator = true_pos + false_pos + true_neg + false_neg
    if denominator == 0:
        prv = (true_pos + false_neg) / (denominator + epsilon)
    else:
        prv = (true_pos + false_neg) / denominator

    return prv


def get_f_score(true_pos, false_pos, false_neg, beta=1, epsilon=1e-8):
    """Get F score, default `beta=1` returns F1 score."""
    denominator = ((1 + beta ** 2) * true_pos +
                   (beta ** 2) * false_pos + false_neg)

    try:
        f_score = ((1 + beta ** 2) * true_pos) / denominator
    except ZeroDivisionError:
        f_score = ((1 + beta ** 2) * true_pos) / (denominator + epsilon)

    return f_score


def get_intersection_over_union(true_pos, false_pos, false_neg, epsilon=1e-8):
    """Get intersection over union."""
    denominator = true_pos + false_pos + false_neg
    if denominator == 0:
        iou = true_pos / (denominator + epsilon)
    else:
        iou = true_pos / denominator

    return iou

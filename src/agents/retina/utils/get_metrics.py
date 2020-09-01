"""Get metrics related objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import logging

from src.metrics import binary_confusion
from src.metrics import area_under_cruves
from src.metrics import dice_coefficient


LOGGER = logging.getLogger(__name__)


def get_metrics(metric_names, prob_maps, binary_maps, targets,
                masks=None):
    """Get metrics values on a batch of probability maps and targets."""
    # incase for latter display
    metrics = collections.OrderedDict()

    # pre-calculate confusion matrix
    if any(metric in ('acc', 'se', 'sp',
                      'f1', 'prc', 'iou') for metric in metric_names):
        (true_pos, false_pos, true_neg,
         false_neg) = binary_confusion.get_binary_confusion_matrix(
             binary_maps, targets, masks, reduction='sum')

        metrics['tp'] = true_pos
        metrics['fp'] = false_pos
        metrics['tn'] = true_neg
        metrics['fn'] = false_neg

    for metric in metric_names:
        if metric == 'acc':
            metrics[metric] = binary_confusion.get_accuracy(
                true_pos, false_pos, true_neg, false_neg)

        elif metric == 'se':
            metrics[metric] = binary_confusion.get_true_positive_rate(
                true_pos, false_neg)

        elif metric == 'sp':
            metrics[metric] = binary_confusion.get_true_negative_rate(
                true_neg, false_pos)

        elif metric == 'f1':
            metrics[metric] = binary_confusion.get_f_score(
                true_pos, false_pos, false_neg)

        elif metric == 'prc':
            metrics[metric] = binary_confusion.get_precision(
                true_pos, false_pos)

        elif metric == 'iou':
            metrics[metric] = binary_confusion.get_intersection_over_union(
                true_pos, false_pos, false_neg)

        elif metric == 'auroc':
            metrics[metric] = area_under_cruves.get_area_under_roc_cruve(
                prob_maps, targets)

        elif metric == 'auprc':
            metrics[metric] = area_under_cruves.get_area_under_pr_cruve(
                prob_maps, targets)

        elif metric == 'ap':
            metrics[metric] = area_under_cruves.get_average_precision_score(
                prob_maps, targets)

        elif metric == 'dice':
            metrics[metric] = dice_coefficient.get_dice_coefficient(
                prob_maps, binary_maps, targets, masks)

        else:
            LOGGER.error('Invalid metric: %s', metric)
            raise NotImplementedError

    return metrics

"""Area under curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score

from src.datasets.utils.check_data_integrity import check_probability_map
from src.datasets.utils.check_data_integrity import check_binary_map
from src.metrics.utils.convert_data import convert_to_ndarray


def get_roc_curve(prob_maps, targets):
    """Get false_positive_rate, true_positive_rate, and coresponding
    thresholds for ROC curve (TPR - FPR curve)."""

    assert check_probability_map(prob_maps)
    assert check_binary_map(targets)

    prob_maps = convert_to_ndarray(prob_maps.flatten())
    targets = convert_to_ndarray(targets.flatten())

    fpr, tpr, thresholds = roc_curve(y_score=prob_maps, y_true=targets)
    return fpr, tpr, thresholds


def get_area_under_roc_cruve(prob_maps, targets):
    """Get area under ROC curve."""

    assert check_probability_map(prob_maps)
    assert check_binary_map(targets)

    prob_maps = convert_to_ndarray(prob_maps.flatten())
    targets = convert_to_ndarray(targets.flatten())

    auroc = roc_auc_score(y_score=prob_maps, y_true=targets)
    return auroc


def get_pr_cruve(prob_maps, targets):
    """Get precision, recall and thresholds for precision - recall curve."""

    assert check_probability_map(prob_maps)
    assert check_binary_map(targets)

    prob_maps = convert_to_ndarray(prob_maps.flatten())
    targets = convert_to_ndarray(targets.flatten())

    precision, recall, thresholds = precision_recall_curve(
        probas_pred=prob_maps, y_true=targets)
    return precision, recall, thresholds


def get_area_under_pr_cruve(prob_maps, targets):
    """Get area under PR curve (precision - recall curve)."""

    precision, recall, _ = get_pr_cruve(prob_maps, targets)
    auprc = auc(x=recall, y=precision)

    return auprc


def get_average_precision_score(prob_maps, targets):
    """Get average precision that summarize PR curve."""

    assert check_probability_map(prob_maps)
    assert check_binary_map(targets)

    prob_maps = convert_to_ndarray(prob_maps.flatten())
    targets = convert_to_ndarray(targets.flatten())

    average_precision = average_precision_score(
        y_score=prob_maps, y_true=targets)
    return average_precision

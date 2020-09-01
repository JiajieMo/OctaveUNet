"""Test metrics module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import random

import torch
import numpy as np
import sklearn

from src.metrics.binary_confusion import get_binary_confusion_matrix
from src.metrics.binary_confusion import get_accuracy
from src.metrics.binary_confusion import get_true_positive_rate
from src.metrics.binary_confusion import get_true_negative_rate
from src.metrics.binary_confusion import get_precision
from src.metrics.binary_confusion import get_prevalence
from src.metrics.binary_confusion import get_intersection_over_union
from src.metrics.binary_confusion import get_f_score

from src.metrics.dice_coefficient import get_dice_coefficient

from src.metrics.area_under_cruves import get_roc_curve
from src.metrics.area_under_cruves import get_area_under_roc_cruve
from src.metrics.area_under_cruves import get_pr_cruve
from src.metrics.area_under_cruves import get_area_under_pr_cruve
from src.metrics.area_under_cruves import get_average_precision_score

from src.metrics.value_meters import AverageMeter
from src.metrics.value_meters import AverageMeters


def get_dummy_data_sample(return_tensor=False):
    """Get dummy data sample for testing binary confusion matrix and
    other metrics based on it."""
    prob_maps = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                          [0.2, 0.3, 0.4, 0.5, 0.6],
                          [0.3, 0.4, 0.5, 0.6, 0.7],
                          [0.4, 0.3, 0.2, 0.1, 0.5],
                          [0.5, 0.4, 0.3, 0.2, 0.1]])[None, None, :, :]
    binary_maps = np.array([[0, 0, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 0, 0, 1],
                            [1, 1, 1, 0, 0]])[None, None, :, :]
    targets = np.array([[0, 0, 0, 1, 1],
                        [0, 0, 1, 1, 0],
                        [0, 1, 1, 1, 1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 0, 0, 0]])[None, None, :, :]

    if return_tensor:
        prob_maps = torch.from_numpy(prob_maps)
        binary_maps = torch.from_numpy(binary_maps)
        targets = torch.from_numpy(targets)

    return prob_maps, binary_maps, targets


class TestBinaryConfusionMetrics(unittest.TestCase):
    """Test metrics based on binary confusion."""

    def test_get_binary_confusion_matrix(self):
        """Test get binary confusion matrix."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, true_neg, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)
        self.assertEqual(true_pos, 10)
        self.assertEqual(false_pos, 5)
        self.assertEqual(true_neg, 8)
        self.assertEqual(false_neg, 2)

    def test_get_accuracy(self):
        """Test get_accuracy."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, true_neg, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)

        acc = get_accuracy(true_pos, false_pos, true_neg, false_neg)
        self.assertEqual(acc, (true_pos + true_neg) /
                         (true_pos + false_pos + true_neg + false_neg))

    def test_get_true_positive_rate(self):
        """Test get_true_positive_rate."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, _, _, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)

        tpr = get_true_positive_rate(true_pos, false_neg)
        self.assertAlmostEqual(tpr, (true_pos) / (true_pos + false_neg))

    def test_get_true_negative_rate(self):
        """Test get_true_negative_rate."""
        _, binary_maps, targets = get_dummy_data_sample()
        _, false_pos, true_neg, _ = get_binary_confusion_matrix(
            binary_maps, targets)

        tnr = get_true_negative_rate(true_neg, false_pos)
        self.assertEqual(tnr, true_neg / (true_neg + false_pos))

    def test_get_precision(self):
        """Test get_precision."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, _, _ = get_binary_confusion_matrix(
            binary_maps, targets)

        prc = get_precision(true_pos, false_pos)
        self.assertEqual(prc, true_pos / (true_pos + false_pos))

    def test_get_prevalence(self):
        """Test get_prevalence."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, true_neg, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)

        prv = get_prevalence(true_pos, false_pos, true_neg, false_neg)
        self.assertEqual(prv, (true_pos + false_neg) /
                         (true_pos + false_pos + true_neg + false_neg))

    def test_get_intersection_over_union(self):
        """Test get_intersection_over_union."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, _, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)

        iou = get_intersection_over_union(true_pos, false_pos, false_neg)
        self.assertEqual(iou, true_pos / (true_pos + false_pos + false_neg))

    def test_get_f_score(self):
        """Test get_f_score."""
        _, binary_maps, targets = get_dummy_data_sample()
        true_pos, false_pos, _, false_neg = get_binary_confusion_matrix(
            binary_maps, targets)

        f1_score = get_f_score(true_pos, false_pos, false_neg, beta=1)
        self.assertEqual(f1_score, (2 * true_pos) /
                         (2 * true_pos + false_neg + false_pos))


class TestDiceCoefficient(unittest.TestCase):
    """Test dice coefficient."""

    def test_get_dice_coefficient(self):
        """Test get binary confusion matrix."""
        prob_maps, binary_maps, targets = get_dummy_data_sample(
            return_tensor=True)
        dice_score = get_dice_coefficient(prob_maps, binary_maps, targets)
        self.assertAlmostEqual(dice_score, 0.6666667)


class TestAreaUnderCruves(unittest.TestCase):
    """Test various metrics of area under cruves."""

    def test_roc_cruve(self):
        """Test get_roc_curve."""
        prob_maps, _, targets = get_dummy_data_sample(return_tensor=True)
        fpr, tpr, thresholds = get_roc_curve(prob_maps, targets)
        auroc = get_area_under_roc_cruve(prob_maps, targets)

        self.assertEqual(len(fpr), len(thresholds))
        self.assertEqual(len(tpr), len(thresholds))
        self.assertEqual(sklearn.metrics.auc(x=fpr, y=tpr), auroc)

    def test_pr_cruve(self):
        """Test precision and recall cruve."""
        prob_maps, _, targets = get_dummy_data_sample(return_tensor=True)
        precision, recall, _ = get_pr_cruve(prob_maps, targets)
        auprc = get_area_under_pr_cruve(prob_maps, targets)
        average_precision = get_average_precision_score(prob_maps, targets)

        self.assertEqual(len(precision), len(recall))
        # not actually equvilent
        self.assertNotEqual(auprc, average_precision)


class TestAverageMeters(unittest.TestCase):
    """Test average meters."""

    def test_average_meter(self):
        """Test average meter."""
        dummy_meter = AverageMeter(name='dummy_data')

        num_samples = 10
        dummy_list = []
        for _ in range(num_samples):
            dummy_value = random.random()
            dummy_list.append(dummy_value)
            dummy_meter.accumulate(dummy_value)

        self.assertEqual(dummy_meter.accumulated_value, np.sum(dummy_list))
        self.assertEqual(dummy_meter.average_value, np.mean(dummy_list))
        self.assertEqual(dummy_meter.current_value, dummy_list[-1])
        self.assertEqual(dummy_meter.accumulate_count, num_samples)
        self.assertEqual(dummy_meter.recorded_values, dummy_list)

        dummy_meter.reset_values()
        self.assertEqual(dummy_meter.average_value, 0.0)

    def test_average_meters(self):
        """Test multiple average meters."""
        names = ['dummy_a', 'dummy_b', 'dummy_c']
        dummy_meters = AverageMeters(names=names)

        num_samples = 10
        dummy_list_of_dict = []
        for _ in range(num_samples):
            dummy_dict = {}
            for name in names:
                dummy_value = random.random()
                dummy_dict[name] = dummy_value

            dummy_meters.accumulate(dummy_dict)
            dummy_list_of_dict.append(dummy_dict)

        for name in names:
            self.assertAlmostEqual(
                dummy_meters.average_values[name],
                np.mean([dummy_dict[name]
                         for dummy_dict in dummy_list_of_dict]),
            )

        dummy_meters.reset_values()
        for name in names:
            self.assertEqual(dummy_meters.average_values[name], 0.0)


if __name__ == "__main__":
    unittest.main()

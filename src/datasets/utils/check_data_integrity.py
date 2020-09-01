"""Check data integrity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def check_binary_map(binary_map):
    """Check integrity of binary_map data."""
    if isinstance(binary_map, torch.Tensor):
        check_unique_fn = torch.unique
    elif isinstance(binary_map, np.ndarray):
        check_unique_fn = np.unique

    cond_a = bool(binary_map.min() == 0)
    cond_b = bool(binary_map.max() == 1)
    cond_c = bool(len(check_unique_fn(binary_map)) == 2)

    cond_d = bool(len(check_unique_fn(binary_map)) == 1)
    cond_e = bool(binary_map.min() == 1)
    cond_f = bool(binary_map.max() == 0)

    case_a = bool(cond_a and cond_b and cond_c)
    case_b = bool(cond_d and (cond_e or cond_f))

    if case_a is True or case_b is True:
        return True

    return False


def check_probability_map(pred):
    """Check integrity of probability binary_map."""
    cond_a = bool(pred.min() >= 0)
    cond_b = bool(pred.max() <= 1)

    if cond_a and cond_b:
        return True

    return False


def check_sample_ids(sample_ids):
    """Check whether `sample_ids` is a iterator of int."""
    cond_a = bool(isinstance(sample_ids, (list, tuple)))
    cond_b = bool(isinstance(item, int) for item in sample_ids)
    if cond_a and cond_b:
        return True

    return False

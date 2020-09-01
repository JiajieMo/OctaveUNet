"""Process control for image processing functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import functools
import random

import numpy as np
import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)


def random_trigger_process(trigger_prob: float = 0.5):
    """Randomly trigger a process."""
    def decorator_random_trigger_process(process):
        @functools.wraps(process)
        def warpper_random_trigger_process(sample, **kwargs):
            updated_trigger_prob = kwargs.get('trigger_prob', trigger_prob)
            assert 0 <= updated_trigger_prob <= 1, 'Trigger probability should be in [0, 1]'
            if random.random() < updated_trigger_prob:
                return process(sample, **kwargs)
            return sample
        return warpper_random_trigger_process
    return decorator_random_trigger_process


def multi_data_process(process):
    """Decorator for process applied on all data of a sample."""
    @functools.wraps(process)
    def warpper_multi_data_process(sample, *args, **kwargs):
        assert isinstance(sample, dict), 'Sample should be a dictionary'
        for sample_key, sample_value in sample.items():
            assert isinstance(
                sample_value, (Image.Image, torch.Tensor, np.ndarray))
            sample[sample_key] = process(sample_value, *args, **kwargs)
        return sample
    return warpper_multi_data_process


def image_only_process(process):
    """Decorator for process applied on image only."""
    @functools.wraps(process)
    def warpper_image_only_process(sample, *args, **kwargs):
        assert isinstance(sample, dict), 'Sample should be a dictionary'

        for sample_key, sample_value in sample.items():
            assert isinstance(
                sample_value, (Image.Image, torch.Tensor, np.ndarray))
            # not binary mask
            case_a = bool(isinstance(sample_value, Image.Image) and
                          sample_value.mode != '1')
            case_b = bool('image' in sample_key)
            if case_a or case_b:
                sample[sample_key] = process(sample_value, *args, **kwargs)
        return sample
    return warpper_image_only_process


def mask_only_process(process):
    """Decorator for process applied on mask only."""
    @functools.wraps(process)
    def warpper_mask_only_process(sample, *args, **kwargs):
        assert isinstance(sample, dict), 'Sample should be a dictionary'
        for sample_key, sample_value in sample.items():
            assert isinstance(
                sample_value, (Image.Image, torch.Tensor, np.ndarray),
                'Sample should be image in PIL, PyTorch, or NumPy format.')
            # not binary mask
            case_a = bool(isinstance(sample_value, Image.Image) and
                          sample_value.mode == '1')
            case_b = bool('image' not in sample_key)
            if case_a or case_b:
                sample[sample_key] = process(sample_value, *args, **kwargs)
        return sample
    return warpper_mask_only_process

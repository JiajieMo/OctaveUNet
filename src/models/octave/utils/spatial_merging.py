"""
Padding into same spatial dimension
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import warnings
from torch.nn import functional as F


def get_padding_pattern(dim_a, dim_b, pad_mode='around'):
    """Get padding pattern from dimensions of two tensors."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', category=torch.jit.TracerWarning, module=r'.*')
        _, _, h_max, w_max = max(dim_a, dim_b)

    w_diff_a = w_max - dim_a[-1]
    h_diff_a = h_max - dim_a[-2]
    w_diff_b = w_max - dim_b[-1]
    h_diff_b = h_max - dim_b[-2]

    if pad_mode == 'around':
        pad_a = (w_diff_a // 2, w_diff_a - w_diff_a // 2,
                 h_diff_a // 2, h_diff_a - h_diff_a // 2)
        pad_b = (w_diff_b // 2, w_diff_b - w_diff_b // 2,
                 h_diff_b // 2, h_diff_b - h_diff_b // 2)
        return pad_a, pad_b

    if pad_mode == 'corner':
        pad_a = (0, w_diff_a, 0, h_diff_a)
        pad_b = (0, w_diff_b, 0, h_diff_b)
        return pad_a, pad_b

    raise NotImplementedError


def padding_into_max_shape(tensor_a, tensor_b, pad_mode='around'):
    """Padding the last two dimensions of tensors into the maximums."""
    pad_a, pad_b = get_padding_pattern(
        tensor_a.shape, tensor_b.shape, pad_mode)

    tensor_a = F.pad(tensor_a, pad=pad_a, mode='constant', value=0)
    tensor_b = F.pad(tensor_b, pad=pad_b, mode='constant', value=0)

    return tensor_a, tensor_b


@torch.jit.script
def slice_helper(batch_images, h_start, h_end, w_start, w_end):
    """Helper function for slicing on tensor of batch of images."""
    return batch_images[:, :, h_start:h_end, w_start:w_end]


def get_cropping_pattern(dim_a, dim_b, crop_mode='around'):
    """Get cropping pattern from dimensions of two tensors."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', category=torch.jit.TracerWarning, module=r'.*')
        _, _, h_min, w_min = min(dim_a, dim_b)

    w_diff_a = dim_a[-1] - w_min
    h_diff_a = dim_a[-2] - h_min
    w_diff_b = dim_b[-1] - w_min
    h_diff_b = dim_b[-2] - h_min

    if crop_mode == 'around':
        w_start_a = w_diff_a // 2
        w_end_a = dim_a[-1] - (w_diff_a - w_diff_a // 2)
        h_start_a = h_diff_a // 2
        h_end_a = dim_a[-2] - (h_diff_a - h_diff_a // 2)

        w_start_b = w_diff_b // 2
        w_end_b = dim_b[-1] - (w_diff_b - w_diff_b // 2)
        h_start_b = h_diff_b // 2
        h_end_b = dim_b[-2] - (h_diff_b - h_diff_b // 2)

        return (h_start_a, h_end_a, w_start_a, w_end_a,
                h_start_b, h_end_b, w_start_b, w_end_b)

    if crop_mode == 'corner':
        w_start_a = 0
        w_end_a = dim_a[-1] - w_diff_a
        h_start_a = 0
        h_end_a = dim_a[-2] - h_diff_a

        w_start_b = 0
        w_end_b = dim_b[-1] - w_diff_b
        h_start_b = 0
        h_end_b = dim_b[-2] - h_diff_b

        return (h_start_a, h_end_a, w_start_a, w_end_a,
                h_start_b, h_end_b, w_start_b, w_end_b)

    raise NotImplementedError


def cropping_into_min_shape(tensor_a, tensor_b, crop_mode='around'):
    """Cropping the last two dimensions of tensors into the minimums."""

    (h_start_a, h_end_a, w_start_a, w_end_a,
     h_start_b, h_end_b, w_start_b, w_end_b) = get_cropping_pattern(
         tensor_a, tensor_b, crop_mode)

    tensor_a = slice_helper(tensor_a, h_start_a, h_end_a, w_start_a, w_end_a)
    tensor_b = slice_helper(tensor_b, h_start_b, h_end_b, w_start_b, w_end_b)

    return tensor_a, tensor_b

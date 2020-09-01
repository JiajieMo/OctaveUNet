"""Utility functions for handling tensor and other common data type."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import torch


def convert_to_ndarray(data):
    """Convert data to numpy ndarray."""
    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        if data.requires_grad is True:
            data = data.detach()

        if data.device != torch.device('cpu'):
            data = data.cpu()

        return data.numpy()

    raise NotImplementedError

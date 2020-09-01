"""Get statistics of dataset of tensor or PIL images."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import logging
from PIL import Image

import numpy as np

import torch
from torchvision.transforms import functional as TF

from src.datasets.utils.check_data_integrity import check_binary_map

LOGGER = logging.getLogger(__name__)


def get_positive_weight(dataset: torch.utils.data.Dataset,
                        target_key: str):
    """Get positive weight."""

    pos_count, neg_count = get_pos_neg_count(dataset, target_key)
    pos_weight = neg_count / pos_count
    pos_weight = torch.from_numpy(np.array(pos_weight))
    LOGGER.debug('Calculated positive weight (positive / negative): %.4f',
                 pos_weight)
    return pos_weight


def get_pos_neg_count(dataset: torch.utils.data.Dataset, target_key: str):
    """Get dataset statistics."""
    pos_count = 0
    neg_count = 0

    for sample in dataset:
        target = sample[target_key]

        if isinstance(target, (Image.Image, np.ndarray)):
            target = TF.to_tensor(target)

        assert check_binary_map(target)

        pos_count += target.sum()
        neg_count += (target.numel() - target.sum())

    pos_count = pos_count.numpy()
    neg_count = neg_count.numpy()

    return pos_count, neg_count


def get_channel_mean_std(dataset: torch.utils.data.Dataset,
                         image_key: str = 'image'):
    """Get channel wise mean and std of a dataset."""
    channel_mean = 0.0
    channel_std = 0.0

    for sample in dataset:
        image = sample[image_key]

        if isinstance(image, (Image.Image, np.ndarray)):
            image = TF.to_tensor(image)

        num_channels, *_ = image.shape

        channel_mean += image.view(num_channels, -1).mean(-1)
        channel_std += image.view(num_channels, -1).std(-1)

    channel_mean /= len(dataset)
    channel_std /= len(dataset)

    channel_mean = channel_mean.numpy().tolist()
    channel_std = channel_std.numpy().tolist()

    return channel_mean, channel_std

"""Image processing utility functions for ONLINE preprocessings of PIL image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
from PIL import Image

from torchvision.transforms import functional as TF
from skimage import morphology
from skimage.color.adapt_rgb import adapt_rgb, each_channel

from src.processings.utils.process_controls import image_only_process
from src.processings.utils.process_controls import multi_data_process

LOGGER = logging.getLogger(__name__)


def resize(sample, **kwargs):
    """Resize all image data."""
    size = kwargs.get('size', (512, 512))
    sample = multi_data_process(lambda img: TF.resize(img, size))(sample)
    return sample


def vessel_enhancement(sample, **kwargs):
    """Apply morphology closing for enhancing vessel structure."""

    # size of structural element, chosen to be 11 so that is bigger than the
    # largest scale of vessels within dataset
    selem_radius = kwargs.get('selem_radius', 11)

    @adapt_rgb(each_channel)
    def channel_closing(img):
        img_closed = morphology.closing(
            img, selem=morphology.disk(selem_radius))
        return img_closed

    def enhancing(img):
        img = np.array(img)
        img_closed = channel_closing(img)
        num_channels = np.shape(img_closed)[-1]
        img_closed_channel_mean = np.reshape(
            img_closed, (-1, num_channels)).mean(0).reshape(1, 1, num_channels)
        img = img - img_closed + img_closed_channel_mean
        return Image.fromarray(img.astype(np.uint8))

    sample = image_only_process(enhancing)(sample)

    return sample


def normalization(sample, **kwargs):
    """Apply channel wise normalize of image data of a sample."""
    channel_mean = kwargs.get('channel_mean', (0.5, 0.5, 0.5))
    channel_std = kwargs.get('channel_std', (0.5, 0.5, 0.5))
    sample = image_only_process(lambda img: TF.to_pil_image(
        TF.normalize(TF.to_tensor(img), channel_mean, channel_std)))(sample)

    return sample

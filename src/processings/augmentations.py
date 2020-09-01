"""Image processing utility functions for data augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
from torchvision.transforms import functional as TF

from src.processings.utils.process_controls import random_trigger_process
from src.processings.utils.process_controls import multi_data_process
from src.processings.utils.process_controls import image_only_process


LOGGER = logging.getLogger(__name__)


@random_trigger_process(trigger_prob=0.5)
@multi_data_process
# pylint: disable=unused-argument
def random_hflip(sample, **kwargs):
    """Randomly apply horizontal flip for all data of a sample."""
    return TF.hflip(sample)


@random_trigger_process(trigger_prob=0.5)
@multi_data_process
# pylint: disable=unused-argument
def random_vflip(sample, **kwargs):
    """Randomly apply verticle flip for all data of a sample."""
    return TF.vflip(sample)


@random_trigger_process(trigger_prob=0.5)
def random_rotate(sample, **kwargs):
    """Randomly apply rotation for all data of a sample."""
    rotate_angle_range = kwargs.get('rotate_angle_range', (-180, 180))
    rotate_angle = random.randint(*rotate_angle_range)
    sample = multi_data_process(
        lambda img: TF.rotate(img, rotate_angle))(sample)
    return sample


@random_trigger_process(trigger_prob=0.5)
def random_adjust_brightness(sample, **kwargs):
    """Randomly apply brightness adjustment for RGB image data of a sample."""
    brightness_factor_range = kwargs.get('brightness_factor_range', (0.8, 1.2))
    brightness_factor = random.uniform(*brightness_factor_range)
    sample = image_only_process(
        lambda img: TF.adjust_brightness(img, brightness_factor))(sample)
    return sample


@random_trigger_process(trigger_prob=0.5)
def random_adjust_contrast(sample, **kwargs):
    """Randomly apply contrast adjustment for RGB image data of a sample."""
    contrast_factor_range = kwargs.get('contrast_factor_range', (0.8, 1.2))
    contrast_factor = random.uniform(*contrast_factor_range)
    sample = image_only_process(
        lambda img: TF.adjust_contrast(img, contrast_factor))(sample)
    return sample


@random_trigger_process(trigger_prob=0.5)
def random_adjust_gamma(sample, **kwargs):
    """Randomly apply gamma adjustment for RGB image data of a sample."""
    gamma_range = kwargs.get('gamma_range', (0.8, 1.2))
    gamma = random.uniform(*gamma_range)
    sample = image_only_process(
        lambda img: TF.adjust_gamma(img, gamma))(sample)
    return sample


@random_trigger_process(trigger_prob=0.5)
def random_adjust_saturation(sample, **kwargs):
    """Randomly apply saturation adjustment for RGB image data of a sample."""
    saturation_factor_range = kwargs.get('saturation_factor_range', (0.8, 1.2))
    saturation_factor = random.uniform(*saturation_factor_range)
    sample = image_only_process(
        lambda img: TF.adjust_saturation(img, saturation_factor))(sample)
    return sample


@random_trigger_process(trigger_prob=0.5)
def random_affine_transform(sample, **kwargs):
    """Randomly apply affine transform for all data of a sample."""
    rotate_angle_range = kwargs.get('rotate_angle_range', (-180, 180))
    translate_range = kwargs.get('translate_range', (0, 0))
    scale_range = kwargs.get('scale_range', (0.9, 1.1))
    shear_range = kwargs.get('shear_range', (-5, 5))
    rotate_angle = random.randint(*rotate_angle_range)
    translate = (random.uniform(*translate_range),
                 random.uniform(*translate_range))
    scale = random.uniform(*scale_range)
    shear = random.randint(*shear_range)

    sample = multi_data_process(lambda img: TF.affine(
        img, rotate_angle, translate, scale, shear))(sample)

    return sample

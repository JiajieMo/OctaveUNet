"""Get model related objects."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from src.configs.config_node import ConfigNode
from src.models.octave.octave_unet import OctaveUNet
from src.models.octave.octave_unet import StaticOctaveUNet


LOGGER = logging.getLogger(__name__)


def get_model(configs: ConfigNode):
    """Get model."""
    model_name = configs.MODEL.MODEL_NAME
    if model_name == 'octave':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'alphas': configs.MODEL.ALPHAS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
            'merge_mode': 'padding',
        }

        model = OctaveUNet(**kwargs)

    elif model_name == 'static_octave':
        kwargs = {
            'channels': configs.MODEL.CHANNELS,
            'alphas': configs.MODEL.ALPHAS,
            'kernel_size': configs.MODEL.KERNEL_SIZE,
            'stride': configs.MODEL.STRIDE,
            'padding': configs.MODEL.PADDING,
            'dilation': configs.MODEL.DILATION,
            'groups': configs.MODEL.GROUPS,
            'bias': configs.MODEL.ENABLE_BIAS,
            'batch_norm': configs.MODEL.ENABLE_BATCH_NORM,
            'dropout': configs.MODEL.ENABLE_DROPOUT,
            'padding_mode': 'zeros',
            'merge_mode': 'padding',
        }

        model = StaticOctaveUNet(**kwargs)

    else:
        LOGGER.error('Invalid model: %s', model_name)
        raise NotImplementedError

    LOGGER.info('Retrieved model: %s', model_name)

    return model

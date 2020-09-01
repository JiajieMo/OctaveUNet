"""Utilities for handling shapes inference of convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from torch.nn.modules.utils import _pair


def get_conv2d_output_shape(input_size, kernel=1, stride=1, padding=0,
                            dilation=1):
    """Get the spatial dimension of output tensor of 2D convolution."""

    input_size = _pair(input_size)
    kernel = _pair(kernel)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    padding = (_pair(padding[0]), _pair(padding[1]))

    height = math.floor((input_size[0] + sum(padding[0]) - dilation[0] *
                         (kernel[0] - 1) - 1) / stride[0] + 1)
    width = math.floor((input_size[1] + sum(padding[1]) - dilation[1] *
                        (kernel[1] - 1) - 1) / stride[1] + 1)

    return height, width


def get_transp2d_output_shape(input_size, kernel=1, stride=1,
                              padding=0, dilation=1, output_padding=0):
    """Get the spatial dimension of output tensor of 2D transposed convolution."""

    input_size = _pair(input_size)
    kernel = _pair(kernel)
    stride = _pair(stride)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    dilation = _pair(dilation)

    padding = (_pair(padding[0]), _pair(padding[1]))

    height = (input_size[0] - 1) * stride[0] - sum(padding[0]) + \
        dilation[0] * (kernel[0] - 1) + output_padding[0] + 1
    width = (input_size[1] - 1) * stride[1] - sum(padding[1]) + \
        dilation[1] * (kernel[1] - 1) + output_padding[1] + 1

    return height, width


def get_conv2d_same_padding(input_size, output_size, kernel=1,
                            stride=1, dilation=1):
    """Get the padding pattern of 2D convolution in order to get the same output
    shape as input shape."""

    input_size = _pair(input_size)
    output_size = _pair(output_size)
    kernel = _pair(kernel)
    stride = _pair(stride)
    dilation = _pair(dilation)

    padding_height = ((output_size[0] - 1) * stride[0] -
                      input_size[0] + dilation[0] * (kernel[0] - 1) + 1)
    padding_width = ((output_size[1] - 1) * stride[1] -
                     input_size[1] + dilation[1] * (kernel[1] - 1) + 1)

    padding_height_right_side = math.floor(padding_height / 2)
    padding_height_left_side = math.ceil(padding_height / 2)

    padding_width_right_side = math.floor(padding_width / 2)
    padding_width_left_side = math.ceil(padding_width / 2)

    return (padding_height_right_side, padding_height_left_side,
            padding_width_right_side, padding_width_left_side)


def get_transp2d_same_padding(input_size, output_size,
                              kernel=1, stride=1, dilation=1,
                              output_padding=0):
    """
    Get the padding pattern of 2D transposed convolution in order to get the
         same output shape as input shape.
    """

    input_size = _pair(input_size)
    output_size = _pair(output_size)
    kernel = _pair(kernel)
    stride = _pair(stride)
    dilation = _pair(dilation)
    output_padding = _pair(output_padding)

    padding_height = -(output_size[0] - 1 - output_padding[0] - dilation[0] * (
        kernel[0] - 1) - (input_size[0] - 1) * stride[0]) / 2
    padding_width = -(output_size[1] - 1 - output_padding[1] - dilation[1] * (
        kernel[1] - 1) - (input_size[1] - 1) * stride[1]) / 2

    padding_height_right_side = math.floor(padding_height / 2)
    padding_height_left_side = math.ceil(padding_height / 2)

    padding_width_right_side = math.floor(padding_width / 2)
    padding_width_left_side = math.ceil(padding_width / 2)

    return (padding_height_right_side, padding_height_left_side,
            padding_width_right_side, padding_width_left_side)

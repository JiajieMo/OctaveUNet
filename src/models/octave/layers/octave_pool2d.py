"""
Octave pooling layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class OctaveAvgPool2d(nn.Module):
    """Octave average pooling."""

    def __init__(self, in_alpha, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True,
                 divisor_override=None):
        super(OctaveAvgPool2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        self.avg_pool_h2h = nn.AvgPool2d(
            kernel_size, stride, padding,
            ceil_mode, count_include_pad,
            divisor_override
        ) if self.high_only or self.multi_res else None

        self.avg_pool_l2l = nn.AvgPool2d(
            kernel_size, stride, padding,
            ceil_mode, count_include_pad,
            divisor_override,
        ) if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.avg_pool_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.avg_pool_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.avg_pool_h2h(input_h)
            input_l = self.avg_pool_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l


class OctaveMaxPool2d(nn.Module):
    """Octave max pooling."""

    def __init__(self, in_alpha, kernel_size, stride=None, padding=0,
                 dilation=1, return_indices=False, ceil_mode=False):
        super(OctaveMaxPool2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        self.max_pool_h2h = nn.MaxPool2d(
            kernel_size, stride, padding,
            dilation, return_indices, ceil_mode
        ) if self.high_only or self.multi_res else None

        self.max_pool_l2l = nn.MaxPool2d(
            kernel_size, stride, padding,
            dilation, return_indices, ceil_mode,
        ) if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ

    def forward(self, input_h, input_l=None):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.avg_pool_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.avg_pool_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.avg_pool_h2h(input_h)
            input_l = self.avg_pool_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l

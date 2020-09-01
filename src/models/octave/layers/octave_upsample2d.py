"""
Octave pooling layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class OctaveUpsample2d(nn.Module):
    """Octave average pooling."""

    def __init__(self, in_alpha, size=None, scale_factor=None, mode='nearest',
                 align_corners=True):
        super(OctaveUpsample2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        upsample_by_scale = size is None and scale_factor is not None
        upsample_by_size = size is not None and scale_factor is None
        assert upsample_by_scale or upsample_by_size

        if self.high_only:
            self.upsample_h2h = nn.Upsample(
                size, scale_factor, mode, align_corners)

        elif self.low_only:
            self.upsample_l2l = nn.Upsample(
                size, scale_factor, mode, align_corners)

        elif self.multi_res:
            self.upsample_h2h = nn.Upsample(
                size, scale_factor, mode, align_corners)

            self.upsample_l2l = nn.Upsample(
                size, scale_factor, mode, align_corners)

        else:
            raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.upsample_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.upsample_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.upsample_h2h(input_h)
            input_l = self.upsample_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l

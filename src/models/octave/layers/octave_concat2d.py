"""
Octave RelU layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

from src.models.octave.utils.spatial_merging import padding_into_max_shape
from src.models.octave.utils.spatial_merging import cropping_into_min_shape


class Concat2d(nn.Module):
    """Concatenation module."""

    def __init__(self, merge_mode='padding'):
        super(Concat2d, self).__init__()
        if merge_mode == 'padding':
            self.merge_shapes = padding_into_max_shape
        elif merge_mode == 'cropping':
            self.merge_shapes = cropping_into_min_shape
        else:
            raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, input_a, input_b):
        input_a, input_b = self.merge_shapes(input_a, input_b)
        input_a = torch.cat((input_a, input_b), dim=1)

        return input_a


class OctaveConcat2d(nn.Module):
    """Octave concatenation."""

    def __init__(self, in_alpha, merge_mode='padding'):
        super(OctaveConcat2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        self.concat = Concat2d(merge_mode)

    # pylint: disable=arguments-differ
    def forward(self, input_h, skip_h, input_l, skip_l):
        if self.low_only:
            assert input_h is None
            assert skip_h is None
            assert input_l is not None
            assert skip_l is not None
            input_l = self.concat(input_l, skip_l)

        elif self.high_only:
            assert input_h is not None
            assert skip_h is not None
            assert input_l is None
            assert skip_l is None
            input_h = self.concat(input_h, skip_h)

        elif self.multi_res:
            assert input_h is not None
            assert skip_h is not None
            assert input_l is not None
            assert skip_l is not None
            input_h = self.concat(input_h, skip_h)
            input_l = self.concat(input_l, skip_l)

        else:
            raise NotImplementedError

        return input_h, input_l

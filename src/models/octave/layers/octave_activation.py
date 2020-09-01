"""
Octave ReLU layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn


class OctaveReLU(nn.Module):
    """Octave ReLU activation."""

    def __init__(self, in_alpha, inplace=False):
        super(OctaveReLU, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)
        self.relu_h2h = nn.ReLU(
            inplace) if self.high_only or self.multi_res else None
        self.relu_l2l = nn.ReLU(
            inplace) if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.relu_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.relu_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.relu_h2h(input_h)
            input_l = self.relu_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l


class OctaveSigmoid(nn.Module):
    """Octave sigmoid activation."""

    def __init__(self, in_alpha):
        super(OctaveSigmoid, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        self.sigmoid_h2h = nn.Sigmoid() if self.high_only or self.multi_res else None
        self.sigmoid_l2l = nn.Sigmoid() if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.sigmoid_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.sigmoid_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.sigmoid_h2h(input_h)
            input_l = self.sigmoid_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l


class OctaveDropout2d(nn.Module):
    """Octave dropout layer."""

    def __init__(self, in_alpha, p=0.1, inplace=False):
        super(OctaveDropout2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        self.dropout_h2h = nn.Dropout2d(
            p, inplace) if self.high_only or self.multi_res else None
        self.dropout_l2l = nn.Dropout2d(
            p, inplace) if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.dropout_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.dropout_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.dropout_h2h(input_h)
            input_l = self.dropout_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l


class OctaveBatchNorm(nn.Module):
    """Octave batch norm layer."""

    def __init__(self, in_channels, in_alpha, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(OctaveBatchNorm, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only = bool(in_alpha == 1)
        self.high_only = bool(in_alpha == 0)
        self.multi_res = bool(0 < in_alpha < 1)

        in_channels_l = int(in_alpha * in_channels)
        in_channels_h = int(in_channels - in_channels_l)

        self.batch_norm_h2h = nn.BatchNorm2d(
            num_features=in_channels_h,
            eps=1e-5,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        ) if self.high_only or self.multi_res else None

        self.batch_norm_l2l = nn.BatchNorm2d(
            num_features=in_channels_l,
            eps=1e-5,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        ) if self.low_only or self.multi_res else None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        if self.low_only:
            assert input_h is None
            assert input_l is not None
            input_l = self.batch_norm_l2l(input_l)

        elif self.high_only:
            assert input_h is not None
            assert input_l is None
            input_h = self.batch_norm_h2h(input_h)

        elif self.multi_res:
            assert input_h is not None
            assert input_l is not None
            input_h = self.batch_norm_h2h(input_h)
            input_l = self.batch_norm_l2l(input_l)

        else:
            raise NotImplementedError

        return input_h, input_l

"""
A blocks of two octave convolution.
"""

from torch import nn
from src.models.octave.layers.octave_conv2d import OctaveConv2d
from src.models.octave.layers.octave_conv2d import OctaveConvTranspose2d
from src.models.octave.layers.octave_activation import OctaveBatchNorm
from src.models.octave.layers.octave_activation import OctaveDropout2d
from src.models.octave.layers.octave_activation import OctaveReLU
from src.models.octave.layers.octave_activation import OctaveSigmoid


class OctaveConvBlock(nn.Module):
    """Octave convolution with batch norm, activation, and dropout."""

    def __init__(self, in_channels, out_channels,
                 in_alpha, out_alpha,
                 batch_norm=True, dropout=False, act_fn=None,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(OctaveConvBlock, self).__init__()

        self.conv = OctaveConv2d(in_channels, out_channels,
                                 in_alpha, out_alpha,
                                 spatial_ratio=spatial_ratio,
                                 merge_mode=merge_mode,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias,
                                 padding_mode=padding_mode)

        if batch_norm is True:
            self.batch_norm = OctaveBatchNorm(in_channels=out_channels,
                                              in_alpha=out_alpha)
        else:
            self.batch_norm = None

        if dropout is True:
            self.dropout = OctaveDropout2d(in_alpha=out_alpha)
        else:
            self.dropout = None

        if act_fn == 'relu':
            self.act_fn = OctaveReLU(in_alpha=out_alpha)

        elif act_fn == 'sigmoid':
            self.act_fn = OctaveSigmoid(in_alpha=out_alpha)

        else:
            self.act_fn = None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        input_h, input_l = self.conv(input_h, input_l)

        if self.batch_norm is not None:
            input_h, input_l = self.batch_norm(input_h, input_l)

        if self.act_fn is not None:
            input_h, input_l = self.act_fn(input_h, input_l)

        if self.dropout is not None:
            input_h, input_l = self.dropout(input_h, input_l)

        return input_h, input_l


class DoubleOctaveConvBlock(nn.Module):
    """Double octave convolution block."""

    def __init__(self, in_channels, mid_channels, out_channels,
                 in_alpha, mid_alpha, out_alpha,
                 batch_norm=True, dropout=False, act_fn=None,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(DoubleOctaveConvBlock, self).__init__()

        self.conv_block_1 = OctaveConvBlock(in_channels, mid_channels,
                                            in_alpha, mid_alpha,
                                            batch_norm=batch_norm,
                                            dropout=dropout,
                                            act_fn=act_fn,
                                            spatial_ratio=spatial_ratio,
                                            merge_mode=merge_mode,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias,
                                            padding_mode=padding_mode)

        self.conv_block_2 = OctaveConvBlock(mid_channels, out_channels,
                                            mid_alpha, out_alpha,
                                            batch_norm=batch_norm,
                                            dropout=dropout,
                                            act_fn=act_fn,
                                            spatial_ratio=spatial_ratio,
                                            merge_mode=merge_mode,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias,
                                            padding_mode=padding_mode)

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        input_h, input_l = self.conv_block_1(input_h, input_l)
        input_h, input_l = self.conv_block_2(input_h, input_l)

        return input_h, input_l


class OctaveConvTransposeBlock(nn.Module):
    """
    Octave transposed convolution with batch norm, activation, and dropout.
    """

    def __init__(self, in_channels, out_channels,
                 in_alpha, out_alpha,
                 batch_norm=True, dropout=False, act_fn=None,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=2, padding=1, output_padding=1,
                 groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(OctaveConvTransposeBlock, self).__init__()

        self.conv_transp = OctaveConvTranspose2d(
            in_channels, out_channels, in_alpha, out_alpha,
            spatial_ratio=spatial_ratio, merge_mode=merge_mode,
            kernel_size=kernel_size, stride=stride,
            padding=padding, output_padding=output_padding,
            groups=groups, bias=bias, dilation=dilation,
            padding_mode=padding_mode
        )

        if batch_norm is True:
            self.batch_norm = OctaveBatchNorm(in_channels=out_channels,
                                              in_alpha=out_alpha)
        else:
            self.batch_norm = None

        if dropout is True:
            self.dropout = OctaveDropout2d(in_alpha=out_alpha)
        else:
            self.dropout = None

        if act_fn == 'relu':
            self.act_fn = OctaveReLU(in_alpha=out_alpha)

        elif act_fn == 'sigmoid':
            self.act_fn = OctaveSigmoid(in_alpha=out_alpha)

        else:
            self.act_fn = None

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        input_h, input_l = self.conv_transp(input_h, input_l)

        if self.batch_norm is not None:
            input_h, input_l = self.batch_norm(input_h, input_l)

        if self.act_fn is not None:
            input_h, input_l = self.act_fn(input_h, input_l)

        if self.dropout is not None:
            input_h, input_l = self.dropout(input_h, input_l)

        return input_h, input_l

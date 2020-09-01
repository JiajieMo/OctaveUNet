"""
Module of a decoder block.
"""

from torch import nn

from src.models.octave.layers.octave_upsample2d import OctaveUpsample2d
from src.models.octave.layers.octave_concat2d import OctaveConcat2d
from src.models.octave.blocks.octave_conv_block import DoubleOctaveConvBlock
from src.models.octave.blocks.octave_conv_block import OctaveConvTransposeBlock


class DecoderBlock(nn.Module):
    """
    Decoder block of octave transposed convolution and double octave convolution.
    """

    def __init__(self, in_channels, mid_channels, out_channels,
                 in_alpha, mid_alpha, out_alpha,
                 upsample='transp', scale_factor=2,
                 batch_norm=True, dropout=False, act_fn=None,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=1, padding=1, output_padding=1,
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DecoderBlock, self).__init__()

        if upsample == 'transp':
            self.upsample = OctaveConvTransposeBlock(
                in_channels, in_channels, in_alpha, in_alpha,
                batch_norm=batch_norm, dropout=dropout, act_fn=act_fn,
                spatial_ratio=spatial_ratio, merge_mode=merge_mode,
                kernel_size=kernel_size, stride=scale_factor,
                padding=padding, output_padding=output_padding,
                groups=groups, bias=bias, dilation=dilation,
                padding_mode=padding_mode
            )

        elif upsample in ('bilinear', 'nearest'):
            self.upsample = OctaveUpsample2d(in_alpha=in_alpha,
                                             scale_factor=scale_factor,
                                             mode=upsample)

        else:
            raise NotImplementedError

        self.oct_concat = OctaveConcat2d(
            in_alpha=in_alpha, merge_mode=merge_mode)

        self.double_conv = DoubleOctaveConvBlock(
            in_channels*2, mid_channels, out_channels,
            in_alpha, mid_alpha, out_alpha,
            batch_norm=batch_norm, dropout=dropout, act_fn=act_fn,
            spatial_ratio=spatial_ratio, merge_mode=merge_mode,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups,
            bias=bias, padding_mode=padding_mode
        )

    # pylint: disable=arguments-differ
    def forward(self, input_h, skip_h, input_l, skip_l):
        input_h, input_l = self.upsample(input_h, input_l)
        input_h, input_l = self.oct_concat(input_h, skip_h, input_l, skip_l)
        input_h, input_l = self.double_conv(input_h, input_l)

        return input_h, input_l

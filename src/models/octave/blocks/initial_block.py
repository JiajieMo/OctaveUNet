"""
Module of the initial block.
"""

from torch import nn

from src.models.octave.blocks.octave_conv_block import DoubleOctaveConvBlock


class InitialBlock(nn.Module):
    """Initial block of 2 octave convolution layers."""

    def __init__(self, **kwargs):
        super(InitialBlock, self).__init__()
        self.in_alpha = kwargs.get('in_alpha')
        assert self.in_alpha in [0, 1]
        self.double_conv = DoubleOctaveConvBlock(**kwargs)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        if self.in_alpha == 0:
            input_h = inputs
            input_l = None
        elif self.in_alpha == 1:
            input_h = None
            input_l = inputs
        else:
            raise NotImplementedError

        input_h, input_l = self.double_conv(input_h, input_l)
        return input_h, input_l

"""
Module of the initial block.
"""

from torch import nn

from src.models.octave.blocks.octave_conv_block import OctaveConvBlock


class FinalBlock(nn.Module):
    """Final block of a octave convolution layers that outputs logits."""

    def __init__(self, **kwargs):
        super(FinalBlock, self).__init__()
        self.out_alpha = kwargs.get('out_alpha')
        assert self.out_alpha in [0, 1]
        assert kwargs.get('act_fn') is None
        self.conv_block = OctaveConvBlock(**kwargs)

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        output_h, output_l = self.conv_block(input_h, input_l)

        if self.out_alpha == 0:
            assert output_l is None
            outputs = output_h

        elif self.out_alpha == 1:
            assert output_h is None
            outputs = output_l

        else:
            raise NotImplementedError

        return outputs

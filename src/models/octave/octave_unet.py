"""
Octave UNet models
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn

from src.models.octave.blocks.initial_block import InitialBlock
from src.models.octave.blocks.encoder_block import EncoderBlock
from src.models.octave.blocks.decoder_block import DecoderBlock
from src.models.octave.blocks.final_block import FinalBlock


class OctaveUNet(nn.Module):
    """Octave UNet with arbitrary layers."""

    def __init__(self, channels, alphas, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, batch_norm=True, dropout=False,
                 padding_mode='zeros', merge_mode='padding'):
        super(OctaveUNet, self).__init__()
        assert len(channels) == len(alphas)
        self.channels = channels
        self.alphas = alphas

        encoder_input_channels = channels[1:-1]
        encoder_output_channels = channels[2:-1] + channels[-2:-1]
        encoder_input_alphas = alphas[1:-1]
        encoder_output_alphas = alphas[2:-1] + alphas[-2:-1]

        decoder_input_channels = channels[-2:0:-1]
        decoder_output_channels = channels[-3:0:-1] + channels[1:2]
        decoder_input_alphas = alphas[-2:0:-1]
        decoder_output_alphas = alphas[-3:0:-1] + alphas[1:2]

        self.add_module('encoder_0',
                        InitialBlock(in_channels=channels[0],
                                     mid_channels=channels[1],
                                     out_channels=channels[1],
                                     in_alpha=alphas[0],
                                     mid_alpha=alphas[1],
                                     out_alpha=alphas[1],
                                     batch_norm=batch_norm,
                                     dropout=dropout,
                                     act_fn='relu',
                                     spatial_ratio=2,
                                     merge_mode=merge_mode,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     dilation=dilation,
                                     groups=groups,
                                     bias=bias,
                                     padding_mode=padding_mode))

        i = 0
        for input_channel, output_channel, input_alpha, output_alpha in zip(
                encoder_input_channels, encoder_output_channels,
                encoder_input_alphas, encoder_output_alphas):
            i += 1
            self.add_module('encoder_{}'.format(i),
                            EncoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         in_alpha=input_alpha,
                                         mid_alpha=output_alpha,
                                         out_alpha=output_alpha,
                                         downsample='avg',
                                         scale_factor=2,
                                         batch_norm=batch_norm,
                                         dropout=dropout,
                                         act_fn='relu',
                                         spatial_ratio=2,
                                         merge_mode=merge_mode,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias,
                                         padding_mode=padding_mode))

        i = len(decoder_input_channels)
        for input_channel, output_channel, input_alpha, output_alpha in zip(
                decoder_input_channels, decoder_output_channels,
                decoder_input_alphas, decoder_output_alphas):
            self.add_module('decoder_{}'.format(i),
                            DecoderBlock(in_channels=input_channel,
                                         mid_channels=output_channel,
                                         out_channels=output_channel,
                                         in_alpha=input_alpha,
                                         mid_alpha=output_alpha,
                                         out_alpha=output_alpha,
                                         upsample='transp',
                                         scale_factor=2,
                                         batch_norm=batch_norm,
                                         dropout=dropout,
                                         act_fn='relu',
                                         spatial_ratio=2,
                                         merge_mode=merge_mode,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=padding,
                                         dilation=dilation,
                                         groups=groups,
                                         bias=bias,
                                         padding_mode=padding_mode))
            i -= 1

        self.add_module('decoder_0',
                        FinalBlock(in_channels=channels[1],
                                   out_channels=channels[-1],
                                   in_alpha=alphas[1],
                                   out_alpha=alphas[-1],
                                   batch_norm=batch_norm,
                                   dropout=dropout,
                                   act_fn=None,
                                   spatial_ratio=2,
                                   merge_mode=merge_mode,
                                   kernel_size=1,
                                   stride=stride,
                                   padding=0,
                                   dilation=dilation,
                                   groups=groups,
                                   bias=bias,
                                   padding_mode=padding_mode))

    # pylint: disable=arguments-differ
    def forward(self, inputs):

        for name, module in self.named_children():
            if name == 'encoder_0':
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(inputs)

            elif name == 'decoder_0':
                outputs = module(locals()[name[:-1] + '1_h'],
                                 locals()[name[:-1] + '1_l'])

            elif name == 'decoder_{}'.format(len(self.channels) - 2):
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                     locals()['en' + name[2:] + '_h'],
                     locals()['en' + name[2:-1] +
                              str(int(name[-1]) - 1) + '_h'],
                     locals()['en' + name[2:] + '_l'],
                     locals()['en' + name[2:-1] + str(
                         int(name[-1]) - 1) + '_l'])

            elif 'encoder_' in name:
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                     locals()[name[:-1] + str(int(name[-1]) - 1) + '_h'],
                     locals()[name[:-1] + str(int(name[-1]) - 1) + '_l'])

            elif 'decoder_' in name:
                (locals()[name + '_h'],
                 locals()[name + '_l']) = module(
                     locals()[name[:-1] + str(int(name[-1]) + 1) + '_h'],
                     locals()['en' + name[2:-1] +
                              str(int(name[-1]) - 1) + '_h'],
                     locals()[name[:-1] + str(int(name[-1]) + 1) + '_l'],
                     locals()['en' + name[2:-1] + str(
                         int(name[-1]) - 1) + '_l'])

        return outputs


class StaticOctaveUNet(nn.Module):
    """Octave UNet with fixed number of layers."""

    def __init__(self, channels, alphas, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1,
                 bias=True, batch_norm=True, dropout=False,
                 padding_mode='zeros', merge_mode='padding'):
        super(StaticOctaveUNet, self).__init__()
        assert len(channels) == 6
        assert len(alphas) == 6

        self.channels = channels
        self.alphas = alphas

        self.encoder_0 = InitialBlock(in_channels=channels[0],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[0],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.encoder_1 = EncoderBlock(in_channels=channels[1],
                                      mid_channels=channels[2],
                                      out_channels=channels[2],
                                      in_alpha=alphas[1],
                                      mid_alpha=alphas[2],
                                      out_alpha=alphas[2],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.encoder_2 = EncoderBlock(in_channels=channels[2],
                                      mid_channels=channels[3],
                                      out_channels=channels[3],
                                      in_alpha=alphas[2],
                                      mid_alpha=alphas[3],
                                      out_alpha=alphas[3],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.encoder_3 = EncoderBlock(in_channels=channels[3],
                                      mid_channels=channels[4],
                                      out_channels=channels[4],
                                      in_alpha=alphas[3],
                                      mid_alpha=alphas[4],
                                      out_alpha=alphas[4],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.encoder_4 = EncoderBlock(in_channels=channels[4],
                                      mid_channels=channels[4],
                                      out_channels=channels[4],
                                      in_alpha=alphas[4],
                                      mid_alpha=alphas[4],
                                      out_alpha=alphas[4],
                                      downsample='avg',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.decoder_4 = DecoderBlock(in_channels=channels[4],
                                      mid_channels=channels[3],
                                      out_channels=channels[3],
                                      in_alpha=alphas[4],
                                      mid_alpha=alphas[3],
                                      out_alpha=alphas[3],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      output_padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.decoder_3 = DecoderBlock(in_channels=channels[3],
                                      mid_channels=channels[2],
                                      out_channels=channels[2],
                                      in_alpha=alphas[3],
                                      mid_alpha=alphas[2],
                                      out_alpha=alphas[2],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      output_padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.decoder_2 = DecoderBlock(in_channels=channels[2],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[2],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      output_padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.decoder_1 = DecoderBlock(in_channels=channels[1],
                                      mid_channels=channels[1],
                                      out_channels=channels[1],
                                      in_alpha=alphas[1],
                                      mid_alpha=alphas[1],
                                      out_alpha=alphas[1],
                                      upsample='transp',
                                      scale_factor=2,
                                      batch_norm=batch_norm,
                                      dropout=dropout,
                                      act_fn='relu',
                                      spatial_ratio=2,
                                      merge_mode=merge_mode,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=padding,
                                      output_padding=padding,
                                      dilation=dilation,
                                      groups=groups,
                                      bias=bias,
                                      padding_mode=padding_mode)

        self.decoder_0 = FinalBlock(in_channels=channels[1],
                                    out_channels=channels[-1],
                                    in_alpha=alphas[1],
                                    out_alpha=alphas[-1],
                                    batch_norm=batch_norm,
                                    dropout=dropout,
                                    act_fn=None,
                                    spatial_ratio=2,
                                    merge_mode=merge_mode,
                                    kernel_size=1,
                                    stride=stride,
                                    padding=0,
                                    dilation=dilation,
                                    groups=groups,
                                    bias=bias,
                                    padding_mode=padding_mode)

    # pylint: disable=arguments-differ
    def forward(self, inputs):
        encoder_0_h, encoder_0_l = self.encoder_0(inputs)
        encoder_1_h, encoder_1_l = self.encoder_1(encoder_0_h, encoder_0_l)
        encoder_2_h, encoder_2_l = self.encoder_2(encoder_1_h, encoder_1_l)
        encoder_3_h, encoder_3_l = self.encoder_3(encoder_2_h, encoder_2_l)
        encoder_4_h, encoder_4_l = self.encoder_4(encoder_3_h, encoder_3_l)
        decoder_4_h, decoder_4_l = self.decoder_4(encoder_4_h, encoder_3_h,
                                                  encoder_4_l, encoder_3_l)
        decoder_3_h, decoder_3_l = self.decoder_3(decoder_4_h, encoder_2_h,
                                                  decoder_4_l, encoder_2_l)
        decoder_2_h, decoder_2_l = self.decoder_2(decoder_3_h, encoder_1_h,
                                                  decoder_3_l, encoder_1_l)
        decoder_1_h, decoder_1_l = self.decoder_1(decoder_2_h, encoder_0_h,
                                                  decoder_2_l, encoder_0_l)
        outputs = self.decoder_0(decoder_1_h, decoder_1_l)

        return outputs

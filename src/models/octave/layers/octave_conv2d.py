"""
Octave convolution layer
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from torch.nn import functional as F

from src.models.octave.utils.spatial_merging import padding_into_max_shape
from src.models.octave.utils.spatial_merging import cropping_into_min_shape


class OctaveConv2d(nn.Module):
    """Octave convolution."""

    def __init__(self, in_channels, out_channels, in_alpha, out_alpha,
                 spatial_ratio=2, merge_mode='padding',
                 upsamle=False, downsample=True,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 downsample_mode='avg', upsample_mode='bilinear'):
        super(OctaveConv2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only_input = bool(in_alpha == 1)
        self.high_only_input = bool(in_alpha == 0)
        self.multi_res_input = bool(0 < in_alpha < 1)

        assert 0 <= out_alpha <= 1
        self.low_only_output = bool(out_alpha == 1)
        self.high_only_output = bool(out_alpha == 0)
        self.multi_res_output = bool(0 < out_alpha < 1)

        in_channels_l = int(in_alpha * in_channels)
        in_channels_h = int(in_channels - in_channels_l)
        out_channels_l = int(out_alpha * out_channels)
        out_channels_h = int(out_channels - out_channels_l)

        self.depth_wise = bool(groups == in_channels)
        groups_h = groups if not self.depth_wise else in_channels_h
        groups_l = groups if not self.depth_wise else in_channels_l

        if self.low_only_input or self.low_only_output:
            self.conv_h2h = None
        else:
            self.conv_h2h = nn.Conv2d(in_channels_h, out_channels_h,
                                      kernel_size, stride, padding, dilation,
                                      groups_h, bias, padding_mode)

        if self.high_only_input or self.high_only_output:
            self.conv_l2l = None
        else:
            self.conv_l2l = nn.Conv2d(in_channels_l, out_channels_l,
                                      kernel_size, stride, padding, dilation,
                                      groups_l, bias, padding_mode)

        if self.low_only_input or self.high_only_output or self.depth_wise:
            self.downsample_h2l = None
            self.conv_h2l = None
        else:
            if downsample_mode == 'avg':
                self.downsample_h2l = nn.AvgPool2d(spatial_ratio)
            elif downsample_mode == 'max':
                self.downsample_h2l = nn.MaxPool2d(spatial_ratio)
            elif downsample_mode == 'conv':
                self.downsample_h2l = nn.Conv2d(in_channels=in_channels_h,
                                                out_channels=in_channels_h,
                                                kernel_size=3,
                                                stride=spatial_ratio,
                                                padding=1,
                                                dilation=1,
                                                groups=1,
                                                bias=True,
                                                padding_mode='zeros')
            else:
                raise NotImplementedError

            self.conv_h2l = nn.Conv2d(in_channels_h, out_channels_l,
                                      kernel_size, stride, padding,
                                      dilation, groups_h, bias,
                                      padding_mode)

        if self.high_only_input or self.low_only_output or self.depth_wise:
            self.conv_l2h = None
            self.upsample_l2h = None
        else:
            self.conv_l2h = nn.Conv2d(in_channels_l, out_channels_h,
                                      kernel_size, stride, padding,
                                      dilation, groups_h, bias,
                                      padding_mode)

            if upsample_mode in ['nearest', 'bilinear']:
                self.upsample_l2h = nn.Upsample(size=None,
                                                scale_factor=spatial_ratio,
                                                mode=upsample_mode,
                                                align_corners=True)

            elif upsample_mode == 'transp':
                self.upsample_l2h = nn.ConvTranspose2d(
                    in_channels=out_channels_h,
                    out_channels=out_channels_h,
                    kernel_size=3,
                    stride=self.spatial_ratio,
                    padding=1,
                    output_padding=1,
                    groups=1,
                    bias=True,
                    dilation=1,
                    padding_mode='zeros')

            else:
                raise NotImplementedError

        if merge_mode == 'padding':
            self.merge_shapes = padding_into_max_shape

        elif merge_mode == 'cropping':
            self.merge_shapes = cropping_into_min_shape

        else:
            raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        try:
            output_h2h = self.conv_h2h(input_h)
        except TypeError:
            output_h2h = None

        # use if/else instead to avoid unnecessary pooling
        if self.conv_h2l is not None:
            output_h2l = self.downsample_h2l(input_h)
            output_h2l = self.conv_h2l(output_h2l)
        else:
            output_h2l = None

        try:
            output_l2l = self.conv_l2l(input_l)
        except TypeError:
            output_l2l = None

        try:
            output_l2h = self.conv_l2h(input_l)
            output_l2h = self.upsample_l2h(output_l2h)

        except TypeError:
            output_l2h = None

        if output_h2h is not None and output_l2h is not None:
            output_h2h, output_l2h = self.merge_shapes(output_h2h, output_l2h)
            output_h = output_h2h + output_l2h

        elif output_h2h is not None and output_l2h is None:
            output_h = output_h2h

        elif output_h2h is None and output_l2h is not None:
            output_h = output_l2h

        else:
            output_h = None

        if output_h2l is not None and output_l2l is not None:
            output_h2l, output_l2l = self.merge_shapes(output_h2l, output_l2l)
            output_l = output_h2l + output_l2l

        elif output_h2l is not None and output_l2l is None:
            output_l = output_h2l

        elif output_h2l is None and output_l2l is not None:
            output_l = output_l2l

        else:
            output_l = None

        return output_h, output_l


class OctaveConvTranspose2d(nn.Module):
    """Octave transposed convolution."""

    def __init__(self, in_channels, out_channels, in_alpha, out_alpha,
                 spatial_ratio=2, merge_mode='padding',
                 kernel_size=3, stride=2, padding=1, output_padding=1,
                 groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(OctaveConvTranspose2d, self).__init__()
        assert 0 <= in_alpha <= 1
        self.low_only_input = bool(in_alpha == 1)
        self.high_only_input = bool(in_alpha == 0)
        self.multi_res_input = bool(0 < in_alpha < 1)

        assert 0 <= out_alpha <= 1
        self.low_only_output = bool(out_alpha == 1)
        self.high_only_output = bool(out_alpha == 0)
        self.multi_res_output = bool(0 < out_alpha < 1)

        in_channels_l = int(in_alpha * in_channels)
        in_channels_h = int(in_channels - in_channels_l)
        out_channels_l = int(out_alpha * out_channels)
        out_channels_h = int(out_channels - out_channels_l)

        self.spatial_ratio = spatial_ratio
        self.depth_wise = bool(groups == in_channels)
        groups_h = groups if not self.depth_wise else in_channels_h
        groups_l = groups if not self.depth_wise else in_channels_l

        if self.low_only_input or self.low_only_output:
            self.conv_transposed_h2h = None
        else:
            self.conv_transposed_h2h = nn.ConvTranspose2d(in_channels_h,
                                                          out_channels_h,
                                                          kernel_size,
                                                          stride,
                                                          padding,
                                                          output_padding,
                                                          groups_h,
                                                          bias,
                                                          dilation,
                                                          padding_mode)

        if self.high_only_input or self.high_only_output:
            self.conv_transposed_l2l = None
        else:
            self.conv_transposed_l2l = nn.ConvTranspose2d(in_channels_l,
                                                          out_channels_l,
                                                          kernel_size,
                                                          stride,
                                                          padding,
                                                          output_padding,
                                                          groups_l,
                                                          bias,
                                                          dilation,
                                                          padding_mode)

        if self.low_only_input or self.high_only_output or self.depth_wise:
            self.conv_transposed_h2l = None
        else:
            self.conv_transposed_h2l = nn.ConvTranspose2d(in_channels_h,
                                                          out_channels_l,
                                                          kernel_size,
                                                          stride,
                                                          padding,
                                                          output_padding,
                                                          groups_h,
                                                          bias,
                                                          dilation,
                                                          padding_mode)

        if self.high_only_input or self.low_only_output or self.depth_wise:
            self.conv_transposed_l2h = None
        else:
            self.conv_transposed_l2h = nn.ConvTranspose2d(in_channels_l,
                                                          out_channels_h,
                                                          kernel_size,
                                                          stride,
                                                          padding,
                                                          output_padding,
                                                          groups_l,
                                                          bias,
                                                          dilation,
                                                          padding_mode)

        if merge_mode == 'padding':
            self.merge_shapes = padding_into_max_shape
        elif merge_mode == 'cropping':
            self.merge_shapes = cropping_into_min_shape
        else:
            raise NotImplementedError

    # pylint: disable=arguments-differ
    def forward(self, input_h, input_l):
        try:
            output_h2h = self.conv_transposed_h2h(input_h)
        except TypeError:
            output_h2h = None

        # use if/else instead to avoid unnecessary pooling
        if self.conv_transposed_h2l is not None:
            output_h2l = F.avg_pool2d(input_h, self.spatial_ratio)
            output_h2l = self.conv_transposed_h2l(output_h2l)
        else:
            output_h2l = None

        try:
            output_l2l = self.conv_transposed_l2l(input_l)
        except TypeError:
            output_l2l = None

        try:
            output_l2h = self.conv_transposed_l2h(input_l)
            # be careful with arguments when calling interpolate
            output_l2h = F.interpolate(
                output_l2h, scale_factor=self.spatial_ratio)
        except TypeError:
            output_l2h = None

        if output_h2h is not None and output_l2h is not None:
            output_h2h, output_l2h = self.merge_shapes(output_h2h, output_l2h)
            output_h = output_h2h + output_l2h

        elif output_h2h is not None and output_l2h is None:
            output_h = output_h2h

        elif output_h2h is None and output_l2h is not None:
            output_h = output_l2h

        else:
            output_h = None

        if output_h2l is not None and output_l2l is not None:
            output_h2l, output_l2l = self.merge_shapes(output_h2l, output_l2l)
            output_l = output_h2l + output_l2l

        elif output_h2l is not None and output_l2l is None:
            output_l = output_h2l

        elif output_h2l is None and output_l2l is not None:
            output_l = output_l2l

        else:
            output_l = None

        return output_h, output_l

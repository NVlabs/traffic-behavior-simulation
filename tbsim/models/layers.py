# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

"""Commonly used network layers and functions."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _pair(x):
    if hasattr(x, '__iter__'):
        return x
    return (x, x)


def _triple(x):
    if hasattr(x, '__iter__'):
        return x
    return (x, x, x)


def init_last_conv_layer(module, b_prior, w_std=0.01):
    """Initializes parameters of a convolutional layer.

    Uses normal distribution for the kernel weights and constant
    computed using bias_prior for the prior.
    See Focal Loss paper for more details.
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.normal_(m.weight, mean=0, std=w_std)
            nn.init.constant_(m.bias, -np.log((1.0 - b_prior) / b_prior))
            break


def soft_threshold(x, thresh, steepness=1024):
    """Returns a soft-thresholded version of the tensor {x}.

    Elements less than {thresh} are set to zero,
    elements appreciably larger than {thresh} are unchanged,
    and elements approaching {thresh} from the positive side
    steeply ramp to zero (faster for higher {steepness}).

    Behavior is very similar to th.threshold(x, thresh, 0),
    but soft_threshold is composed of operations that are supported
    by both the PyTorch 1.4 ONNX exporter and the TRT 6 ONNX parser.

    For comparison, th.threshold with nonzero thresh / value
    cannot be exported by PyTorch 1.4, and other formulas
    (like x * (x > thresh).float()) use operations
    which are not supported by the TensorRT 6 importer.
    """
    return x * ((x - thresh) * steepness).clamp(0, 1)


# noqa pylint: disable=R0911
def Activation(act='elu'):
    """Create activation function."""
    if act == 'elu':
        return nn.ELU(inplace=True)
    if act == 'smooth_elu':
        return SmoothELU()
    if act == 'sigmoid':
        return nn.Sigmoid()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'relu':
        return nn.ReLU(inplace=True)
    if act == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    if act is not None:
        raise ValueError('Activation is not supported: {}.'.format(act))
    return nn.Sequential()


def Normalization2D(norm, num_features):
    """Create 2D layer normalization (4D inputs)."""
    if norm == 'bn':
        return nn.BatchNorm2d(num_features)
    if norm is not None:
        raise ValueError('Normalization is not supported: {}.'.format(norm))
    return nn.Sequential()


def Normalization3D(norm, num_features):
    """Create 3D layer normalization (5D inputs)."""
    if norm == 'bn':
        return nn.BatchNorm3d(num_features)
    if norm is not None:
        raise ValueError('Normalization is not supported: {}.'.format(norm))
    return nn.Sequential()


def Upsample2D(mode, in_channels, out_channels, kernel_size, stride,
               padding, output_padding):
    """Create upsampling layer for 4D input (2D convolution).

    Currently 3 interpolation modes are suported: interp, interp2, and deconv.
    interp mode uses nearest neighbor interpolation + convolution.
    interp2 mode also uses nearest neighbor interpolation + convolution,
      but with a custom implementation that performs better in TRT 5 / 6.
    deconv mode uses transposed convolution.
    """

    if stride == 1:
        # Skip all upsampling
        mode = 'interp'

    if mode == 'interp':
        return UpsampleConv2D(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, output_padding=output_padding,
                              use_pth_interp=True)
    if mode == 'interp2':
        return UpsampleConv2D(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, output_padding=output_padding,
                              use_pth_interp=False)
    if mode == 'deconv':
        layer = nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size=stride + 2 * (stride // 2), stride=stride,
                                   padding=stride//2, output_padding=0)

        # fix initialization to not have checkerboarding
        with torch.no_grad():
            layer.weight.data[:] = layer.weight.data[:, :, :1, :1] * 0.5
            layer.weight.data[:, :, 1:-1, 1:-1] *= 2
        return layer
    raise ValueError('Mode is not supported: {}'.format(mode))


def Upsample3D(mode, in_channels, out_channels, kernel_size, stride,
               padding, output_padding):
    """Create upsampling layer for 5D input (3D convolution).

    Currently 2 interpolation modes are suported: interp and deconv.
    interp mode uses nearest neighbor interpolation + convolution.
    deconv mode uses transposed convolution.
    """

    if mode == 'interp':
        return UpsampleConv3D(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, output_padding=output_padding)
    if mode == 'deconv':
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, output_padding=output_padding)

    raise ValueError('Mode is not supported: {}'.format(mode))


class SmoothELU(nn.Module):
    """Smooth version of ELU-like activation function.

    ELU derivative is continuous but not smooth.
    See Improved Training of WGANs paper for more details.
    """

    def forward(self, x):
        """Forward pass."""
        return F.softplus(2.0 * x + 2.0) / 2.0 - 1.0


class UpsampleConv2D(nn.Module):
    """Upsampling that uses nearest neighbor + 2D convolution."""

    def __init__(self,  in_channels, out_channels, kernel_size, stride,
                 padding, output_padding, use_pth_interp=True):
        """Creates the upsampler.

        use_pth_interp forces using PyTorch interpolation (torch.nn.functional.interpolate)
        when applicable, rather than custom interpolation implementation.
        """
        super(UpsampleConv2D, self).__init__()

        stride = _pair(stride)
        if len(stride) != 2:
            raise ValueError('Stride must be either int or 2-tuple but got {}'.format(stride))
        if stride[0] != stride[1]:
            raise ValueError('H and W strides must be equal but got {}'.format(stride))

        self._scale_factor = stride[0]
        self._use_pth_interp = use_pth_interp

        self.interp = self.interpolation(self._scale_factor) if self._scale_factor > 1 else None
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, # noqa: disable=E221
                                stride=1, padding=padding)

    def interpolation(self, scale_factor=2, mode='nearest'):
        """Returns interpolation module."""
        if self._use_pth_interp:
            return nn.Upsample(scale_factor=scale_factor, mode=mode)

        return self._int_upsample

    def _int_upsample(self, x):
        """Alternative implementation of nearest-neighbor interpolation.

        The main motivation is suboptimal performance of TRT NN interpolation (as of TRT 5/6)
        for certain tensor dimensions. The implementation below is about 50% faster
        than the current TRT implementation on V100/FP16.
        """
        assert x.dim() == 4, 'Expected NCHW tensor but got {}.'.format(x.size())
        # TODO(akamenev)[TRT-hack]: due to lack of support of some features (e.g. Tile op)
        # in TRT have to limit to a basic case of 2x upsample.
        assert self._scale_factor == 2, 'Only scale factor == 2 is currently supported' \
            ' but got {}.'.format(self._scale_factor)

        n, c, h, w = [int(d) for d in x.size()]
        # 1. Upsample in W dim.
        # # TODO(akamenev)[TRT-hack] the code below upsets TRT ONNX parser.
        # x = x.view(-1, 1).repeat(1, self._scale_factor)
        x = x.reshape(n, c, -1, 1)
        # Note: when using interp2, Alfred TRT 6.2 ONNX parser requires 4D tensors.
        x = torch.cat((x, x), dim=-1)
        # 2. Upsample in H dim.
        x = x.reshape(n, c, h, w * 2)
        # y = x.repeat(1, 1, 1, self._scale_factor)
        y = torch.cat((x, x), dim=3)
        y = y.reshape(n, c, h * 2, w * 2)
        return y

    def forward(self, x):
        """Forward pass."""
        res = self.interp(x) if self.interp is not None else x
        return self.conv(res)


class UpsampleConv3D(nn.Module):
    """Upsampling that uses nearest neighbor + 3D convolution."""

    def __init__(self,  in_channels, out_channels, kernel_size, stride,
                 padding, output_padding):
        """Creates the upsampler."""
        super(UpsampleConv3D, self).__init__()

        stride = _pair(stride)
        if len(stride) != 3:
            raise ValueError('Stride must be either int or 3-tuple but got {}'.format(stride))
        if stride[0] != 1:
            raise ValueError('Upsampling in D dimension is not supported ({}).'.format(stride))
        if stride[1] != stride[2]:
            raise ValueError('H and W strides must be equal but got {}'.format(stride))

        self.interp = self.interpolation(stride[1]) if stride[1] > 1 else None
        self.conv   = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, # noqa: disable=E221
                                stride=1, padding=padding)

    def interpolation(self, scale_factor=2, mode='nearest'):
        """Returns interpolation module."""
        return nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        """Forward pass."""
        res = x
        if self.interp is not None:
            res = self.interp(x.view(x.size(0), x.size(1) * x.size(2), x.size(3), x.size(4)))
            res = res.view(res.size(0), x.size(1), x.size(2), res.size(-2), res.size(-1))
        return self.conv(res)


class ASPPBlock(nn.Module):
    """Adaptive spatial pyramid pooling block."""

    def __init__(self, in_channels, out_channels, k_hw=3, norm=None, act='relu',
                 dilation_rates=(1, 6, 12), act_before_combine=False, act_after_combine=True,
                 combine_type='add', combine_conv_k_hw=(3,)):
        """Creates the ASPP block."""
        super().__init__()

        # Layers with varying dilation rate
        self._aspp_layers = nn.ModuleList()
        for rate in dilation_rates:
            self._aspp_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=k_hw,
                              padding=rate * (k_hw // 2), dilation=(rate, rate)),
            )

        # Optional norm / activation before combining
        if act_before_combine:
            self._before_combine = nn.Sequential(
                Normalization2D(norm, out_channels),
                Activation(act),
            )
        else:
            self._before_combine = nn.Sequential()

        self._combine_type = combine_type

        # Layers to apply after combining
        post_combine_layers = []
        if combine_type == 'concat':
            assert len(combine_conv_k_hw) > 0
            post_combine_layers.extend([
                nn.Conv2d(
                    out_channels * len(dilation_rates),
                    out_channels,
                    kernel_size=combine_conv_k_hw[0],
                    padding=combine_conv_k_hw[0] // 2,
                ),
                Normalization2D(norm, out_channels),
                Activation(act),
            ])
            combine_conv_k_hw = combine_conv_k_hw[1:]

        for c_c_k_hw in combine_conv_k_hw:
            post_combine_layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=c_c_k_hw, padding=c_c_k_hw // 2),
                Normalization2D(norm, out_channels),
                Activation(act),
            ])
        if not act_after_combine and post_combine_layers:
            # Remove the final activation
            post_combine_layers = post_combine_layers[:-1]

        self._final_layer = nn.Sequential(*post_combine_layers)

    def _combine_op(self, xs):
        if self._combine_type == 'concat':
            res = torch.cat(xs, -3)
        elif self._combine_type == 'add':
            res = xs[0]
            for x in xs[1:]:
                res = res + x
        else:
            raise ValueError('Combine type {} is not supported.'.format(self._combine_type))
        return res

    def forward(self, x):
        """Forward pass."""
        aspp_outs = []
        for layer in self._aspp_layers:
            aspp_outs.append(self._before_combine(layer(x)))
        return self._final_layer(self._combine_op(aspp_outs))


class ResnetBlock2D(nn.Module):
    """Residual block with 2D convolutions.

    Supports both basic and bottleneck configurations.
    """

    def __init__(self, in_channels, dim, out_channels, k_hw=3, s_hw=1,
                 bottleneck=True, norm='bn', act='relu'):
        """Creates the residual block."""
        super(ResnetBlock2D, self).__init__()

        if k_hw < 3 or (k_hw % 2) == 0:
            raise ValueError('ResnetBlock2D requires kernel size '
                             'to be odd and >= 3 but got {}'.format(k_hw))

        self.act = Activation(act)
        layers = []
        p_hw = k_hw // 2
        if bottleneck:
            layers += [
                # 1x1 squeeze.
                nn.Conv2d(in_channels, dim, kernel_size=1, stride=1, padding=0),
                Normalization2D(norm, dim),
                Activation(act),
                # 3x3 (or kxk)
                nn.Conv2d(dim, dim, kernel_size=k_hw, stride=s_hw, padding=p_hw),
                Normalization2D(norm, dim),
                Activation(act),
                # 1x1 expand.
                nn.Conv2d(dim, out_channels, kernel_size=1, stride=1, padding=0),
                Normalization2D(norm, out_channels)]
            self.block = nn.Sequential(*layers)
        else:
            layers += [
                # First 3x3 (or kxk).
                nn.Conv2d(in_channels, dim, kernel_size=k_hw, stride=s_hw, padding=p_hw),
                Normalization2D(norm, dim),
                Activation(act),
                # Second 3x3 (or kxk).
                nn.Conv2d(dim, out_channels, kernel_size=k_hw, stride=1, padding=p_hw),
                Normalization2D(norm, out_channels)]
            self.block = nn.Sequential(*layers)

        self.shortcut = None
        if in_channels != out_channels or s_hw > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=s_hw, padding=0),
                Normalization2D(norm, out_channels))

    def forward(self, x):
        """Forward pass."""
        res = self.block(x)
        res += x if self.shortcut is None else self.shortcut(x)
        return self.act(res)


class ResnetBlockTran2D(nn.Module):
    """Transposed residual block with 2D convolutions.

    Supports both basic and bottleneck configurations.
    """

    def __init__(self, in_channels, dim, out_channels, k_hw=3, s_hw=1,
                 bottleneck=True, norm='bn', act='relu', upsample='interp'):
        """Creates the transposed residual block."""
        super(ResnetBlockTran2D, self).__init__()

        if k_hw < 3 or (k_hw % 2) == 0:
            raise ValueError('ResnetBlockTran2D requires kernel size '
                             'to be odd and >= 3 but got {}'.format(k_hw))

        self.act = Activation(act)
        layers = []
        p_hw = k_hw // 2
        o_p_hw = p_hw if s_hw > 1 else 0
        if bottleneck:
            layers += [
                # 1x1 squeeze.
                nn.Conv2d(in_channels, dim, kernel_size=1, stride=1, padding=0),
                Normalization2D(norm, dim),
                Activation(act),
                # 3x3 (or kxk)
                Upsample2D(upsample, dim, dim, kernel_size=k_hw, stride=s_hw,
                           padding=p_hw, output_padding=o_p_hw),
                Normalization2D(norm, dim),
                Activation(act),
                # 1x1 expand.
                nn.Conv2d(dim, out_channels, kernel_size=1, stride=1, padding=0),
                Normalization2D(norm, out_channels)]
            self.block = nn.Sequential(*layers)
        else:
            layers += [
                # First 3x3 (or kxk).
                Upsample2D(upsample, in_channels, dim, kernel_size=k_hw, stride=s_hw,
                           padding=p_hw, output_padding=o_p_hw),
                Normalization2D(norm, dim),
                Activation(act),
                # Second 3x3 (or kxk).
                nn.Conv2d(dim, out_channels, kernel_size=k_hw, stride=1, padding=p_hw),
                Normalization2D(norm, out_channels)]
            self.block = nn.Sequential(*layers)

        self.shortcut = None
        if in_channels != out_channels or s_hw > 1:
            self.shortcut = nn.Sequential(
                Upsample2D(upsample, in_channels, out_channels, kernel_size=1, stride=s_hw,
                           padding=0, output_padding=o_p_hw),
                Normalization2D(norm, out_channels))

    def forward(self, x):
        """Forward pass."""
        res = self.block(x)
        res += x if self.shortcut is None else self.shortcut(x)
        return self.act(res)


class ResnetBlock3D(nn.Module):
    """Residual block with 3D convolutions.

    Supports both basic and bottleneck configurations.
    """

    def __init__(self, in_channels, dim, out_channels,
                 k_hw=3, k_d=1, s_hw=1, s_d=1,
                 bottleneck=True, norm='bn', act='relu'):
        """Creates the residual block."""
        super(ResnetBlock3D, self).__init__()

        self.act = Activation(act)
        layers = []
        p_hw   = k_hw // 2 # noqa: disable=E221
        if bottleneck:
            layers += [
                # 1x1 squeeze.
                nn.Conv3d(in_channels, dim, kernel_size=1, stride=1, padding=0),
                Normalization3D(norm, dim),
                self.act,
                # 3x3 (or kxk)
                nn.Conv3d(dim, dim, kernel_size=(k_d, k_hw, k_hw), stride=(s_d, s_hw, s_hw),
                          padding=(0, p_hw, p_hw)),
                Normalization3D(norm, dim),
                self.act,
                # 1x1 expand.
                nn.Conv3d(dim, out_channels, kernel_size=1, stride=1, padding=0),
                Normalization3D(norm, out_channels)]
            self.block = nn.Sequential(*layers)
        else:
            layers += [
                # First 3x3 (or kxk).
                nn.Conv3d(in_channels, dim, kernel_size=(k_d, k_hw, k_hw), stride=(s_d, s_hw, s_hw),
                          padding=(0, p_hw, p_hw)),
                Normalization3D(norm, dim),
                self.act,
                # Second 3x3 (or kxk).
                nn.Conv3d(dim, out_channels, kernel_size=(k_d, k_hw, k_hw), stride=1,
                          padding=(0, p_hw, p_hw)),
                Normalization3D(norm, out_channels)]
            self.block = nn.Sequential(*layers)

        self.shortcut = None
        if in_channels != out_channels or s_hw > 1 or s_d > 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=(s_d, s_hw, s_hw),
                          padding=0),
                Normalization3D(norm, out_channels))

    def forward(self, x):
        """Forward pass."""
        res  = self.block(x) # noqa: disable=E221
        res += x if self.shortcut is None else self.shortcut(x)
        return self.act(res)


class ResnetBlockTran3D(nn.Module):
    """Transposed residual block with 3D convolutions.

    Supports both basic and bottleneck configurations.
    """

    def __init__(self, in_channels, dim, out_channels,
                 k_hw=3, k_d=1, s_hw=1, s_d=1,
                 bottleneck=True, norm='bn', act='relu', upsample='interp'):
        """Creates the transposed residual block."""
        super(ResnetBlockTran3D, self).__init__()

        self.act = Activation(act)
        layers = []
        p_hw   = k_hw // 2 # noqa: disable=E221
        o_p_hw = p_hw if s_hw > 1 else 0
        if bottleneck:
            layers += [
                # 1x1 squeeze.
                nn.Conv3d(in_channels, dim, kernel_size=1, stride=1, padding=0),
                Normalization3D(norm, dim),
                self.act,
                # 3x3 (or kxk)
                Upsample3D(upsample, dim, dim, kernel_size=(k_d, k_hw, k_hw),
                           stride=(s_d, s_hw, s_hw),
                           padding=(0, p_hw, p_hw), output_padding=(0, o_p_hw, o_p_hw)),
                Normalization3D(norm, dim),
                self.act,
                # 1x1 expand.
                nn.Conv3d(dim, out_channels, kernel_size=1, stride=1, padding=0),
                Normalization3D(norm, out_channels)]
            self.block = nn.Sequential(*layers)
        else:
            layers += [
                # First 3x3 (or kxk).
                Upsample3D(upsample, in_channels, dim, kernel_size=(k_d, k_hw, k_hw),
                           stride=(s_d, s_hw, s_hw),
                           padding=(0, p_hw, p_hw), output_padding=(0, o_p_hw, o_p_hw)),
                Normalization3D(norm, dim),
                self.act,
                # Second 3x3 (or kxk).
                nn.Conv3d(dim, out_channels, kernel_size=(k_d, k_hw, k_hw), stride=1,
                          padding=(0, p_hw, p_hw)),
                Normalization3D(norm, out_channels)]
            self.block = nn.Sequential(*layers)

        self.shortcut = None
        if in_channels != out_channels or s_hw > 1 or s_d > 1 or k_d > 1:
            self.shortcut = nn.Sequential(
                Upsample3D(upsample, in_channels, out_channels, kernel_size=(k_d, 1, 1),
                           stride=(s_d, s_hw, s_hw),
                           padding=(0, 0, 0), output_padding=(0, o_p_hw, o_p_hw)),
                Normalization3D(norm, out_channels))

    def forward(self, x):
        """Forward pass."""
        res  = self.block(x) # noqa: disable=E221
        res += x if self.shortcut is None else self.shortcut(x)
        return self.act(res)

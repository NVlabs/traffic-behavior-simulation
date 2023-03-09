import numpy as np
import math
import textwrap
from collections import OrderedDict
from typing import Dict, Union, List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import RoIAlign
from tbsim.models.Transformer import SimpleTransformer

from tbsim.utils.tensor_utils import reshape_dimensions, flatten
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics


class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_dim (int): dimension of outputs
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert(len(dropouts) == len(layer_dims))
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = str(self.__class__.__name__)
        act = None if self._act is None else self._act.__name__
        output_act = None if self._output_act is None else self._output_act.__name__

        indent = ' ' * 4
        msg = "input_dim={}\noutput_shape={}\nlayer_dims={}\nlayer_func={}\ndropout={}\nact={}\noutput_act={}".format(
            self._input_dim, self.output_shape(), self._layer_dims,
            self._layer_func.__name__, self._dropouts, act, output_act
        )
        msg = textwrap.indent(msg, indent)
        msg = header + '(\n' + msg + '\n)'
        return msg


class SplitMLP(MLP):
    """
    A multi-output MLP network: The model split and reshapes the output layer to the desired output shapes
    """

    def __init__(
            self,
            input_dim: int,
            output_shapes: OrderedDict,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_shapes (dict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """

        assert isinstance(output_shapes, OrderedDict)
        output_dim = 0
        for v in output_shapes.values():
            output_dim += np.prod(v)
        self._output_shapes = output_shapes

        super(SplitMLP, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def output_shape(self, input_shape=None):
        return self._output_shapes

    def forward(self, inputs):
        outs = super(SplitMLP, self).forward(inputs)
        out_dict = dict()
        ind = 0
        for k, v in self._output_shapes.items():
            v_dim = int(np.prod(v))
            out_dict[k] = reshape_dimensions(
                outs[:, ind: ind + v_dim], begin_axis=1, end_axis=2, target_dims=v)
            ind += v_dim
        return out_dict


class MIMOMLP(SplitMLP):
    """
    A multi-input, multi-output MLP: The model flattens and concatenate the input before feeding into an MLP
    """

    def __init__(
            self,
            input_shapes: OrderedDict,
            output_shapes: OrderedDict,
            layer_dims: tuple = (),
            layer_func=nn.Linear,
            layer_func_kwargs=None,
            activation=nn.ReLU,
            dropouts=None,
            normalization=False,
            output_activation=None,
    ):
        """
        Args:
            input_shapes (OrderedDict): named dictionary of input shapes
            output_shapes (OrderedDict): named dictionary of output shapes
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        assert isinstance(input_shapes, OrderedDict)
        input_dim = 0
        for v in input_shapes.values():
            input_dim += np.prod(v)

        self._input_shapes = input_shapes

        super(MIMOMLP, self).__init__(
            input_dim=input_dim,
            output_shapes=output_shapes,
            layer_dims=layer_dims,
            layer_func=layer_func,
            layer_func_kwargs=layer_func_kwargs,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation
        )

    def forward(self, inputs):
        flat_inputs = []
        for k in self._input_shapes.keys():
            flat_inputs.append(flatten(inputs[k]))
        return super(MIMOMLP, self).forward(torch.cat(flat_inputs, dim=1))


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # (Mohit): argh... forgot to remove this batchnorm
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # (Mohit): argh... forgot to remove this batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        U-Net forward by concatenating input feature (x1) with mirroring encoding feature maps channel-wise (x2)
        Args:
            x1 (torch.Tensor): [B, C1, H1, W1]
            x2 (torch.Tensor): [B, C2, H2, W2]

        Returns:
            output (torch.Tensor): [B, out_channels, H2, W2]
        """
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(IdentityBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            filters3) if self.batchnorm else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        if self.final_relu:
            out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=kernel_size, dilation=1,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(
            filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class UNetDecoder(nn.Module):
    """UNet part based on https://github.com/milesial/Pytorch-UNet/tree/master/unet"""

    def __init__(self, input_channel, output_channel, encoder_channels, up_factor, bilinear=True, batchnorm=True):
        super(UNetDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 1024, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )

        self.up1 = Up(1024 + encoder_channels[-1], 512, bilinear=True)

        self.up2 = Up(512 + encoder_channels[-2], 512 // up_factor, bilinear)

        self.up3 = Up(256 + encoder_channels[-3], 256 // up_factor, bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3,
                      stride=1, batchnorm=batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3,
                          stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3,
                      stride=1, batchnorm=batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3,
                          stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3,
                      stride=1, batchnorm=batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3,
                          stride=1, batchnorm=batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, output_channel, kernel_size=1)
        )

    def forward(self, feat_to_decode: torch.Tensor, encoder_feats: List[torch.Tensor], target_hw: tuple):
        assert len(encoder_feats) >= 3
        x = self.conv1(feat_to_decode)
        x = self.up1(x, encoder_feats[-1])
        x = self.up2(x, encoder_feats[-2])
        x = self.up3(x, encoder_feats[-3])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        x = F.interpolate(
            x, size=(target_hw[0], target_hw[1]), mode='bilinear')
        return x


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
            self,
            input_shape,
            num_kp=None,
            temperature=1.,
            learnable_temperature=False,
            output_variance=False,
            noise_std=0.0,
    ):
        """
        Args:
            input_shape (list, tuple): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter('temperature', temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(
                torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer('temperature', temperature)

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self._in_w),
            np.linspace(-1., 1., self._in_h)
        )
        pos_x = torch.from_numpy(pos_x.reshape(
            1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            1, self._in_h * self._in_w)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

        self.kps = None

    def __repr__(self):
        """Pretty print network."""
        header = format(str(self.__class__.__name__))
        return header + '(num_kp={}, temperature={}, noise={})'.format(
            self._num_kp, self.temperature.item(), self.noise_std)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)
        assert(input_shape[0] == self._in_c)
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert(feature.shape[1] == self._in_c)
        assert(feature.shape[2] == self._in_h)
        assert(feature.shape[3] == self._in_w)
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(
                self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(
                self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(
                self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat(
                [var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        if isinstance(feature_keypoints, tuple):
            self.kps = (feature_keypoints[0].detach(),
                        feature_keypoints[1].detach())
        else:
            self.kps = feature_keypoints.detach()
        return feature_keypoints


class RasterizedMapEncoder(nn.Module):
    """A basic image-based rasterized map encoder"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            feature_dim: int = None,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None,
            output_activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.model_arch = model_arch
        self.num_input_channels = input_image_shape[0]
        self._feature_dim = feature_dim
        if output_activation is None:
            self._output_activation = nn.Identity()
        else:
            self._output_activation = output_activation()

        # configure conv backbone
        if model_arch == "resnet18":
            self.map_model = resnet18()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (512, out_h, out_w)
        elif model_arch == "resnet50":
            self.map_model = resnet50()
            out_h = int(math.ceil(input_image_shape[1] / 32.))
            out_w = int(math.ceil(input_image_shape[2] / 32.))
            self.conv_out_shape = (2048, out_h, out_w)
        else:
            raise NotImplementedError(f"Model arch {model_arch} unknown")

        # configure spatial reduction pooling layer
        if use_spatial_softmax:
            pooling = SpatialSoftmax(
                input_shape=self.conv_out_shape, **spatial_softmax_kwargs)
            self.pool_out_dim = int(
                np.prod(pooling.output_shape(self.conv_out_shape)))
        else:
            pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.pool_out_dim = self.conv_out_shape[0]

        self.map_model.conv1 = nn.Conv2d(
            self.num_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.map_model.avgpool = pooling
        if feature_dim is not None:
            self.map_model.fc = nn.Linear(
                in_features=self.pool_out_dim, out_features=feature_dim)
        else:
            self.map_model.fc = nn.Identity()

    def output_shape(self, input_shape=None):
        if self._feature_dim is not None:
            return [self._feature_dim]
        else:
            return [self.pool_out_dim]

    def feature_channels(self):
        if self.model_arch in ["resnet18", "resnet34"]:
            channels = OrderedDict({
                "layer1": 64,
                "layer2": 128,
                "layer3": 256,
                "layer4": 512,
            })
        else:
            channels = OrderedDict({
                "layer1": 256,
                "layer2": 512,
                "layer3": 1024,
                "layer4": 2048,
            })
        return channels

    def feature_scales(self):
        return OrderedDict({
            "layer1": 1/4,
            "layer2": 1/8,
            "layer3": 1/16,
            "layer4": 1/32
        })

    def forward(self, map_inputs) -> torch.Tensor:
        feat = self.map_model(map_inputs)
        feat = self._output_activation(feat)
        return feat


class RotatedROIAlign(nn.Module):
    def __init__(self, roi_feature_size, roi_scale):
        super(RotatedROIAlign, self).__init__()
        from tbsim.models.roi_align import ROI_align
        self.roi_align = lambda feat, rois: ROI_align(
            feat, rois, roi_feature_size[0])
        self.roi_scale = roi_scale

    def forward(self, feats, list_of_rois):
        scaled_rois = []
        for rois in list_of_rois:
            sroi = rois.clone()
            sroi[..., :-1] = rois[..., :-1] * self.roi_scale
            scaled_rois.append(sroi)
        list_of_feats = self.roi_align(feats, scaled_rois)
        return torch.cat(list_of_feats, dim=0)


class RasterizeROIEncoder(nn.Module):
    """Use RoI Align to crop map feature for each agent"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            agent_feature_dim: int = None,
            global_feature_dim: int = None,
            roi_feature_size: tuple = (7, 7),
            roi_layer_key: str = "layer4",
            output_activation=nn.ReLU,
            use_rotated_roi=True
    ) -> None:
        super(RasterizeROIEncoder, self).__init__()
        encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=global_feature_dim,
        )
        feat_nodes = {
            'map_model.layer1': 'layer1',
            'map_model.layer2': 'layer2',
            'map_model.layer3': 'layer3',
            'map_model.layer4': 'layer4',
            'map_model.fc': "final"
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)

        self.roi_layer_key = roi_layer_key
        roi_scale = encoder.feature_scales()[roi_layer_key]
        roi_channel = encoder.feature_channels()[roi_layer_key]
        if use_rotated_roi:
            self.roi_align = RotatedROIAlign(
                roi_feature_size, roi_scale=roi_scale)
        else:
            self.roi_align = RoIAlign(
                output_size=roi_feature_size,
                spatial_scale=roi_scale,
                sampling_ratio=-1,
                aligned=True
            )

        self.activation = output_activation()
        if agent_feature_dim is not None:
            self.agent_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),  # [B, C, 1, 1]
                nn.Flatten(start_dim=1),
                nn.Linear(roi_channel, agent_feature_dim),
                self.activation
            )
        else:
            self.agent_net = nn.Identity()

    def forward(self, map_inputs: torch.Tensor, rois: torch.Tensor):
        """

        Args:
            map_inputs (torch.Tensor): [B, C, H, W]
            rois (torch.Tensor): [num_boxes, 5]

        Returns:
            agent_feats (torch.Tensor): [num_boxes, ...]
            roi_feats (torch.Tensor): [num_boxes, ...]
            global_feats (torch.Tensor): [B, ....]
        """
        feats = self.encoder_heads(map_inputs)
        pre_roi_feats = feats[self.roi_layer_key]
        roi_feats = self.roi_align(pre_roi_feats, rois)  # [num_boxes, C, H, W]
        agent_feats = self.agent_net(roi_feats)
        global_feats = feats["final"]

        return agent_feats, roi_feats, self.activation(global_feats), feats


class RasterizedMapKeyPointNet(RasterizedMapEncoder):
    """Predict a list of keypoints [x, y] given a rasterized map"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            spatial_softmax_kwargs=None
    ) -> None:
        super().__init__(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=None,
            use_spatial_softmax=True,
            spatial_softmax_kwargs=spatial_softmax_kwargs
        )

    def forward(self, map_inputs) -> torch.Tensor:
        outputs = super(RasterizedMapKeyPointNet, self).forward(map_inputs)
        # reshape back to kp format
        return outputs.reshape(outputs.shape[0], -1, 2)


class RasterizedMapUNet(nn.Module):
    """Predict a spatial map same size as the input rasterized map"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape: tuple = (3, 224, 224),
            output_channel=1,
            use_spatial_softmax=False,
            spatial_softmax_kwargs=None
    ) -> None:
        super(RasterizedMapUNet, self).__init__()
        encoder = RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            use_spatial_softmax=use_spatial_softmax,
            spatial_softmax_kwargs=spatial_softmax_kwargs
        )
        self.input_image_shape = input_image_shape
        # build graph for extracting intermediate features
        feat_nodes = {
            'map_model.layer1': 'layer1',
            'map_model.layer2': 'layer2',
            'map_model.layer3': 'layer3',
            'map_model.layer4': 'layer4',
        }
        self.encoder_heads = create_feature_extractor(encoder, feat_nodes)
        encoder_channels = list(encoder.feature_channels().values())
        self.decoder = UNetDecoder(
            input_channel=encoder_channels[-1],
            encoder_channels=encoder_channels[:-1],
            output_channel=output_channel,
            up_factor=2
        )

    def forward(self, map_inputs, encoder_feats=None):
        if encoder_feats is None:
            encoder_feats = self.encoder_heads(map_inputs)
        encoder_feats = [encoder_feats[k] for k in ["layer1", "layer2", "layer3", "layer4"]]
        return self.decoder.forward(
            feat_to_decode=encoder_feats[-1],
            encoder_feats=encoder_feats[:-1],
            target_hw=self.input_image_shape[1:]
        )


class RNNTrajectoryEncoder(nn.Module):
    def __init__(self, trajectory_dim, rnn_hidden_size, feature_dim=None, mlp_layer_dims: tuple = (),mode="last"):
        super(RNNTrajectoryEncoder, self).__init__()
        self.mode= mode
        self.lstm = nn.LSTM(
            trajectory_dim, hidden_size=rnn_hidden_size, batch_first=True)
        if feature_dim is not None:
            self.mlp = MLP(
                input_dim=rnn_hidden_size,
                output_dim=feature_dim,
                layer_dims=mlp_layer_dims,
                output_activation=nn.ReLU
            )
            self._feature_dim = feature_dim
        else:
            self.mlp = nn.Identity()
            self._feature_dim = rnn_hidden_size

    def output_shape(self, input_shape=None):
        return [self._feature_dim]

    def forward(self, input_trajectory):
        if self.mode=="last":
            traj_feat = self.lstm(input_trajectory)[0][:, -1, :]
        elif self.mode=="all":
            traj_feat = self.lstm(input_trajectory)[0]
        traj_feat = self.mlp(traj_feat)
        return traj_feat


class RNNFeatureRoller(nn.Module):
    def __init__(self, trajectory_dim, feature_dim):
        super(RNNFeatureRoller, self).__init__()
        self.gru = nn.GRU(
            trajectory_dim, hidden_size=feature_dim, batch_first=True)
        self._feature_dim = feature_dim

    def output_shape(self, input_shape=None):
        return [self._feature_dim]

    def forward(self, feature,input_trajectory):
        _,hn = self.gru(input_trajectory,feature.unsqueeze(0))
        return hn[0]+feature


class PosteriorEncoder(nn.Module):
    """Posterior Encoder (x, x_c -> q) for CVAE"""

    def __init__(
            self,
            condition_dim: int,
            trajectory_shape: tuple,  # [T, D]
            output_shapes: OrderedDict,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100
    ) -> None:
        super(PosteriorEncoder, self).__init__()
        self.trajectory_shape = trajectory_shape

        # TODO: history encoder
        self.traj_encoder = RNNTrajectoryEncoder(
            trajectory_dim=trajectory_shape[-1],
            rnn_hidden_size=rnn_hidden_size
        )
        self.mlp = SplitMLP(
            input_dim=(rnn_hidden_size + condition_dim),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs, condition_features) -> Dict[str, torch.Tensor]:
        traj_feat = self.traj_encoder(inputs["trajectories"])
        feat = torch.cat((traj_feat, condition_features), dim=-1)
        return self.mlp(feat)

class ScenePosteriorEncoder(nn.Module):
    """Scene Posterior Encoder (x, x_c -> q) for CVAE"""

    def __init__(
            self,
            condition_dim: int,
            trajectory_shape: tuple,  # [T, D]
            output_shapes: OrderedDict,
            aggregate_func="max",
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100
    ) -> None:
        super(ScenePosteriorEncoder, self).__init__()
        self.trajectory_shape = trajectory_shape
        self.transformer = SimpleTransformer(src_dim=rnn_hidden_size + condition_dim)
        self.aggregate_func = aggregate_func

        # TODO: history encoder
        self.traj_encoder = RNNTrajectoryEncoder(
            trajectory_dim=trajectory_shape[-1],
            rnn_hidden_size=rnn_hidden_size
        )
        self.mlp = SplitMLP(
            input_dim=(rnn_hidden_size + condition_dim),
            output_shapes=output_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs, condition_features, mask, pos) -> Dict[str, torch.Tensor]:
        bs,Na,T = inputs["trajectories"].shape[:3]

        traj_feat = self.traj_encoder(TensorUtils.join_dimensions(inputs["trajectories"],0,2))
        feat = torch.cat((traj_feat, TensorUtils.join_dimensions(condition_features,0,2)), dim=-1).reshape(bs,Na,-1)

        feat = self.transformer(feat,mask,pos)+feat
        if self.aggregate_func=="max":
            feat = feat.max(1)[0]
        elif self.aggregate_func=="mean":
            feat = feat.mean(1)

        return self.mlp(feat)


class ConditionEncoder(nn.Module):
    """Condition Encoder (x -> c) for CVAE"""

    def __init__(
            self,
            map_encoder: nn.Module,
            trajectory_shape: tuple,  # [T, D]
            condition_dim: int,
            mlp_layer_dims: tuple = (128, 128),
            goal_encoder=None,
            agent_traj_encoder=None
    ) -> None:
        super(ConditionEncoder, self).__init__()
        self.map_encoder = map_encoder
        self.trajectory_shape = trajectory_shape
        self.goal_encoder = goal_encoder
        self.agent_traj_encoder = agent_traj_encoder

        # TODO: history encoder
        # self.hist_lstm = nn.LSTM(trajectory_shape[-1], hidden_size=rnn_hidden_size, batch_first=True)
        visual_feature_size = self.map_encoder.output_shape()[0]
        self.mlp = MLP(
            input_dim=visual_feature_size,
            output_dim=condition_dim,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, condition_inputs):
        map_feat = self.map_encoder(condition_inputs["image"])
        c_feat = self.mlp(map_feat)
        if self.goal_encoder is not None:
            goal_feat = self.goal_encoder(condition_inputs["goal"])
            c_feat = torch.cat([c_feat, goal_feat], dim=-1)
        if self.agent_traj_encoder is not None and "agent_traj" in condition_inputs:
            agent_traj_feat = self.agent_traj_encoder(condition_inputs["agent_traj"])
            c_feat = torch.cat([c_feat, agent_traj_feat], dim=-1)
        return c_feat


class ECEncoder(nn.Module):
    """Condition Encoder (x -> c) for CVAE"""

    def __init__(
            self,
            map_encoder,
            trajectory_shape: tuple,  # [T, D]
            EC_dim: int,
            condition_dim: int, 
            mlp_layer_dims: tuple = (128, 128),
            goal_encoder=None,
            rnn_hidden_size: int = 100
    ) -> None:
        super(ECEncoder, self).__init__()
        if isinstance(map_encoder,nn.Module):
            self.map_encoder = map_encoder
            visual_feature_size = self.map_encoder.output_shape()[0]
        elif isinstance(map_encoder,int):
            visual_feature_size = map_encoder
            self.map_encoder = None
        self.trajectory_shape = trajectory_shape
        self.goal_encoder = goal_encoder
        self.EC_dim = EC_dim
        # TODO: history encoder
        # self.hist_lstm = nn.LSTM(trajectory_shape[-1], hidden_size=rnn_hidden_size, batch_first=True)
        self.traj_encoder = RNNTrajectoryEncoder(trajectory_shape[-1], rnn_hidden_size, feature_dim=EC_dim)
        goal_dim = 0 if goal_encoder is None else goal_encoder.output_shape()[0]
        self.mlp = MLP(
            input_dim=visual_feature_size+goal_dim+EC_dim,
            output_dim=condition_dim,
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, condition_inputs):
        if self.map_encoder is None:
            c_feat = condition_inputs["map_feature"]
        else:
            c_feat = self.map_encoder(condition_inputs["image"])
        if self.goal_encoder is not None:
            goal_feat = self.goal_encoder(condition_inputs["goal"])
            c_feat = torch.cat([c_feat, goal_feat], dim=-1)
        if "cond_traj" in condition_inputs and condition_inputs["cond_traj"] is not None:
            if condition_inputs["cond_traj"].ndim==3:
                EC_feat = self.traj_encoder(condition_inputs["cond_traj"]).unsqueeze(1)
            elif condition_inputs["cond_traj"].ndim==4:
                bs,M,T,D = condition_inputs["cond_traj"].shape
                EC_feat = self.traj_encoder(condition_inputs["cond_traj"].reshape(-1,T,D)).reshape(bs,M,-1)
                
                EC_feat = EC_feat*(condition_inputs["cond_traj"][...,0]!=0).any(-1).unsqueeze(-1)
                EC_feat = torch.cat((EC_feat,torch.zeros([bs,1,self.EC_dim]).to(EC_feat.device)),1)
        else:
            bs = c_feat.shape[0]
            EC_feat = torch.zeros([bs,1,self.EC_dim]).to(c_feat.device)
        if c_feat.ndim==2:
            c_feat = c_feat.unsqueeze(1).repeat(1,EC_feat.shape[1],1)
        else: 
            assert c_feat.ndim==3 and c_feat.shape[1]==EC_feat.shape[1]
        c_feat = torch.cat((c_feat,EC_feat),-1)
        c_feat = self.mlp(c_feat)
        
        return c_feat


class AgentTrajEncoder(nn.Module):
    """Condition Encoder (x -> c) for CVAE"""

    def __init__(
            self,
            trajectory_shape: tuple,  # [T, D]
            feature_dim: int,
            mlp_layer_dims: tuple = (128, 128),
            rnn_hidden_size: int = 100,
            use_transformer = True,
    ) -> None:
        super(AgentTrajEncoder, self).__init__()
        self.trajectory_shape = trajectory_shape
        self.feature_dim = feature_dim
        self.traj_encoder = RNNTrajectoryEncoder(trajectory_shape[-1], rnn_hidden_size, feature_dim=feature_dim, mlp_layer_dims=mlp_layer_dims)
        if use_transformer:
            self.transformer = SimpleTransformer(
                src_dim=feature_dim)
        else:
            self.transformer = None

    def forward(self, agent_trajs):
        bs,M,T,D = agent_trajs.shape
        feat = self.traj_encoder(agent_trajs.reshape(-1,T,D)).reshape(bs,M,-1)
        if self.transformer is not None:
            agent_pos = agent_trajs[:,:,0,:2]
            avails = (agent_pos!=0).any(-1)
            feat = self.transformer(feat,avails,agent_pos)
        
        return torch.max(feat,1)[0]


class PosteriorNet(nn.Module):
    def __init__(
            self,
            input_shapes: OrderedDict,
            condition_dim: int,
            param_shapes: OrderedDict,
            mlp_layer_dims: tuple = ()
    ):
        super(PosteriorNet, self).__init__()
        all_shapes = deepcopy(input_shapes)
        all_shapes["condition_features"] = (condition_dim,)
        self.mlp = MIMOMLP(
            input_shapes=all_shapes,
            output_shapes=param_shapes,
            layer_dims=mlp_layer_dims,
            output_activation=None
        )

    def forward(self, inputs: dict, condition_features: torch.Tensor):
        all_inputs = dict(inputs)
        all_inputs["condition_features"] = condition_features
        return self.mlp(all_inputs)


class ConditionNet(nn.Module):
    def __init__(
            self,
            condition_input_shapes: OrderedDict,
            condition_dim: int,
            mlp_layer_dims: tuple = ()
    ):
        super(ConditionNet, self).__init__()
        self.mlp = MIMOMLP(
            input_shapes=condition_input_shapes,
            output_shapes=OrderedDict(feat=(condition_dim,)),
            layer_dims=mlp_layer_dims,
            output_activation=nn.ReLU
        )

    def forward(self, inputs: dict):
        return self.mlp(inputs)["feat"]


class ConditionDecoder(nn.Module):
    """Decoding (z, c) -> x' using a flat MLP"""

    def __init__(self, decoder_model: nn.Module):
        super(ConditionDecoder, self).__init__()
        self.decoder_model = decoder_model

    def forward(self, latents, condition_features, **decoder_kwargs):
        return self.decoder_model(torch.cat((latents, condition_features), dim=-1), **decoder_kwargs)


class TrajectoryDecoder(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            network_kwargs: dict = None,
            Gaussian_var = False
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
            Gaussian_var (bool): whether output the variance of the predicted trajectory
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self._network_kwargs = network_kwargs
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self.Gaussian_var = Gaussian_var
        self._create_dynamics()
        self._create_networks()

    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        elif self._dynamics_type in ["Bicycle", dynamics.DynType.BICYCLE]:
            self.dyn = dynamics.Bicycle(
                acc_bound=self._dynamics_kwargs["acce_bound"],
                ddh_bound=self._dynamics_kwargs["ddh_bound"],
                max_hdot=self._dynamics_kwargs["max_yawvel"],
                max_speed=self._dynamics_kwargs["max_speed"]
            )
        else:
            self.dyn = None

    def _create_networks(self):
        raise NotImplementedError

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        raise NotImplementedError

    def _forward_dynamics(self, current_states, actions,**kwargs):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        bound = True if "predict" in kwargs and kwargs["predict"] else False
        x, pos, yaw = self.dyn.forward_dynamics(
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time,
            bound = bound,
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj,x

    def forward(self, inputs, current_states=None, num_steps=None,**kwargs):
        preds = self._forward_networks(
            inputs, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            preds["controls"] = preds["trajectories"]
            preds["trajectories"], x = self._forward_dynamics(
                current_states=current_states,
                actions=preds["trajectories"],
                **kwargs
            )
            preds["terminal_state"] = x[...,-1,:]
        return preds


class MLPTrajectoryDecoder(TrajectoryDecoder):
    def _create_networks(self):
        net_kwargs = dict() if self._network_kwargs is None else dict(self._network_kwargs)
        if self._network_kwargs is None:
            net_kwargs = dict()
        
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.dyn.udim))
        if self.Gaussian_var:
            pred_shapes["logvar"] = (self.num_steps, self.state_dim)

        state_as_input = net_kwargs.pop("state_as_input")
        if self.dyn is not None:
            assert state_as_input   # TODO: deprecated, set default to True and remove from configs

        if state_as_input and self.dyn is not None:
            feature_dim = self.feature_dim + self.dyn.xdim
        else:
            feature_dim = self.feature_dim

        self.mlp = SplitMLP(
            input_dim=feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, current_states=None, num_steps=None):
        if self._network_kwargs["state_as_input"] and self.dyn is not None:
            inputs = torch.cat((inputs, current_states), dim=-1)

        if inputs.ndim == 2:
            # [B, D]
            preds = self.mlp(inputs)
        elif inputs.ndim == 3:
            # [B, A, D]
            preds = TensorUtils.time_distributed(inputs, self.mlp)
        else:
            raise ValueError(
                "Expecting inputs to have ndim == 2 or 3, got {}".format(inputs.ndim))
        return preds


class MLPECTrajectoryDecoder(TrajectoryDecoder):
    def __init__(
            self,
            feature_dim: int,
            state_dim: int = 3,
            num_steps: int = None,
            dynamics_type: Union[str, dynamics.DynType] = None,
            dynamics_kwargs: dict = None,
            step_time: float = None,
            EC_feature_dim = 64,
            network_kwargs: dict = None,
            Gaussian_var = False,
            
    ):
        """
        A class that predict future trajectories based on input features
        Args:
            feature_dim (int): dimension of the input feature
            state_dim (int): dimension of the output trajectory at each step
            num_steps (int): (optional) number of future state to predict
            dynamics_type (str, dynamics.DynType): (optional) if specified, the network predicts action
                for the dynamics model instead of future states. The actions are then used to predict
                the future trajectories.
            step_time (float): time between steps. required for using dynamics models
            network_kwargs (dict): keyword args for the decoder networks
            Gaussian_var (bool): whether output the variance of the predicted trajectory
        """
        super(TrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.step_time = step_time
        self.EC_feature_dim = EC_feature_dim
        self._network_kwargs = network_kwargs
        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self.Gaussian_var = Gaussian_var
        self._create_dynamics()
        self._create_networks()

    def _create_networks(self):
        net_kwargs = dict() if self._network_kwargs is None else dict(self._network_kwargs)
        if self._network_kwargs is None:
            net_kwargs = dict()
        
        assert isinstance(self.num_steps, int)
        if self.dyn is None:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.state_dim))
        else:
            pred_shapes = OrderedDict(
                trajectories=(self.num_steps, self.dyn.udim))
        if self.Gaussian_var:
            pred_shapes["logvar"] = (self.num_steps, self.state_dim)

        state_as_input = net_kwargs.pop("state_as_input")
        if self.dyn is not None:
            assert state_as_input   # TODO: deprecated, set default to True and remove from configs

        if state_as_input and self.dyn is not None:
            feature_dim = self.feature_dim + self.dyn.xdim
        else:
            feature_dim = self.feature_dim

        self.mlp = SplitMLP(
            input_dim=feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )
        self.offsetmlp = SplitMLP(
            input_dim=feature_dim+self.EC_feature_dim,
            output_shapes=pred_shapes,
            output_activation=None,
            **net_kwargs
        )

    def _forward_networks(self, inputs, EC_feat=None, current_states=None, num_steps=None):
        if self._network_kwargs["state_as_input"] and self.dyn is not None:
            inputs = torch.cat((inputs, current_states), dim=-1)
        if inputs.ndim == 2:
            # [B, D]
            
            preds = self.mlp(inputs)
            if EC_feat is not None:
                bs,M = EC_feat.shape[:2]

                # EC_feat = self.traj_encoder(cond_traj.reshape(-1,T,D)).reshape(bs,M,-1)
                inputs_tile = inputs.unsqueeze(1).tile(1,M,1)
                EC_feat = torch.cat((inputs_tile,EC_feat),dim=-1)
                EC_preds = TensorUtils.time_distributed(EC_feat, self.offsetmlp)
                EC_preds["trajectories"] = EC_preds["trajectories"] + preds["trajectories"].unsqueeze(1)
            else:
                EC_preds = None

        elif inputs.ndim == 3:
            # [B, A, D]
            preds = TensorUtils.time_distributed(inputs, self.mlp)
            if EC_feat is not None:
                assert EC_feat.ndim==4
                bs,A,M= EC_feat.shape[:3]
                # EC_feat = self.traj_encoder(cond_traj.reshape(-1,T,D)).reshape(bs,M,-1)
                inputs_tile = inputs.tile(1,M,1)
                EC_feat = torch.cat((inputs_tile,EC_feat),dim=-1)
                EC_preds = TensorUtils.time_distributed(EC_feat, self.offsetmlp)
                EC_preds = reshape_dimensions(EC_preds,1,2,(A,M))
                EC_preds["trajectories"] = EC_preds["trajectories"] + preds["trajectories"].unsqueeze(2)
            else:
                EC_preds = None
        else:
            raise ValueError(
                "Expecting inputs to have ndim == 2 or 3, got {}".format(inputs.ndim))
        return preds, EC_preds
    
    def _forward_dynamics(self, current_states, actions,**kwargs):
        assert self.dyn is not None
        assert current_states.shape[-1] == self.dyn.xdim
        assert actions.shape[-1] == self.dyn.udim
        assert isinstance(self.step_time, float) and self.step_time > 0
        bound = True if "predict" in kwargs and kwargs["predict"] else False
        _, pos, yaw = self.dyn.forward_dynamics(
            initial_states=current_states,
            actions=actions,
            step_time=self.step_time,
            bound = bound,
        )
        traj = torch.cat((pos, yaw), dim=-1)
        return traj
    def forward(self, inputs, current_states=None, EC_feat=None, num_steps=None,**kwargs):
        preds, EC_preds = self._forward_networks(
            inputs, EC_feat, current_states=current_states, num_steps=num_steps)
        if self.dyn is not None:
            preds["controls"] = preds["trajectories"]
            if EC_preds is None:
                preds["trajectories"] = self._forward_dynamics(
                    current_states=current_states,
                    actions=preds["trajectories"],
                    **kwargs
                )
            else:
                total_actions = torch.cat((preds["trajectories"].unsqueeze(1),EC_preds["trajectories"]),1)
                bs,A,T,D = total_actions.shape
                current_states_tiled = current_states.unsqueeze(1).repeat(1,total_actions.size(1),1)
                total_trajectories = self._forward_dynamics(
                    current_states=current_states_tiled.reshape(bs*A,-1),
                    actions=total_actions.reshape(bs*A,T,D),
                    **kwargs
                ).reshape(*total_actions.shape[:-1],-1)
                preds["trajectories"] = total_trajectories[:,0]
                preds["EC_trajectories"] = total_trajectories[:,1:]
        else:
            preds["EC_trajectories"] = EC_preds["trajectories"]
                
        return preds



class clique_gibbs_distr(nn.Module):
    def __init__(self,state_enc_dim,edge_encoding_dim,z_dim,edge_types,node_types,node_hidden_dim=[64,64],edge_hidden_dim = [64,64]):
        super(clique_gibbs_distr, self).__init__()
        self.edge_encoding_dim = edge_encoding_dim
        self.z_dim = z_dim
        self.state_enc_dim = state_enc_dim
        self.node_types = node_types
        self.edge_types = edge_types
        self.et_name=dict()
        for et in self.edge_types:
            self.et_name[et]=et[0].name+'->'+et[1].name

        self.node_factor = nn.ModuleDict()
        self.edge_factor = nn.ModuleDict()
        for node_type in self.node_types:
            self.node_factor[node_type.name] = MLP(state_enc_dim[node_type],z_dim,node_hidden_dim)

        for edge_type in self.edge_types:
            self.edge_factor[self.et_name[edge_type]] = MLP(edge_encoding_dim[edge_type],z_dim**2,edge_hidden_dim)
    def forward(self,node_types,node_enc,edge_enc,clique_index, clique_node_index, clique_order_index, clique_is_robot=None):
        device = node_enc.device
        bs,Na = node_types.shape[:2]
        Nc,max_cs = clique_node_index.shape[1:]
        
        # clique_edge_mask = (clique_index.unsqueeze(1)==clique_index.unsqueeze(2))*(clique_index>=0).unsqueeze(1)
        # clique_edge_mask = clique_edge_mask*(~torch.eye(Na,dtype=torch.bool,device=device).unsqueeze(0))
        node_factor = torch.zeros([bs,Na,self.z_dim]).to(device)
        for nt in self.node_types:
            
            node_factor += self.node_factor[nt.name](node_enc)*(node_types==nt.value).unsqueeze(-1)
        
        edge_factor = torch.zeros([bs,Na,Na,self.z_dim**2]).to(device)
        for et in self.edge_types:
            et_flag = (node_types==et[0].value).unsqueeze(1) * (node_types==et[1].value).unsqueeze(2)
            edge_factor +=self.edge_factor[self.et_name[et]](edge_enc[...,-1,:]) * et_flag.unsqueeze(-1)
        edge_factor = edge_factor.reshape(bs,Na,Na,self.z_dim,self.z_dim)

        

        
        node_factor_extended = torch.cat((node_factor,torch.zeros_like(node_factor[:,0:1])),1)
        edge_factor_extended = torch.cat(
            (torch.cat((edge_factor,torch.zeros([bs,1,Na,self.z_dim,self.z_dim],device=device)),1),
            torch.zeros([bs,Na+1,1,self.z_dim,self.z_dim],device=device)),2
        )
        edge_factor_extended_flat = edge_factor_extended.reshape(bs,-1,self.z_dim,self.z_dim)             


            
        logpi = torch.zeros([bs,Nc,*(max_cs*[self.z_dim])]).to(device)
        z = torch.stack(torch.meshgrid(*(max_cs*[torch.arange(self.z_dim)])),-1).to(device)
        
        # add the node factor to the Gibbs distribution
        idx = clique_node_index.reshape(bs,-1).unsqueeze(-1).repeat_interleave(self.z_dim,-1)
        idx = idx.masked_fill(idx==-1,Na)
        node_factor_clique = torch.gather(node_factor_extended,1,idx).reshape(bs,Nc,max_cs,self.z_dim)
        for i in range(max_cs):
            logpi+=node_factor_clique[...,i,:].reshape([bs,Nc,*(i*[1]),self.z_dim,*((max_cs-i-1)*[1])])
        
        # add the edge factor to the Gibbs distribution
        for i in range(max_cs-1):
            for j in range(i+1,max_cs):
                idx = clique_node_index[:,:,[i,j]]
                idx = idx.masked_fill(idx==-1,Na)
                idx_flat = idx[...,0]*(Na+1)+idx[...,1]
                idx_flat = idx_flat.reshape(bs,-1,1,1).repeat_interleave(self.z_dim,2).repeat_interleave(self.z_dim,3)
                edge_factor_ij = torch.gather(edge_factor_extended_flat,1,idx_flat)
                logpi += edge_factor_ij.reshape(bs,Nc,*(i*[1]),self.z_dim,*((j-i-1)*[1]),self.z_dim,*((max_cs-j-1)*[1]))   

        return logpi, z

class AdditiveAttention(nn.Module):
    # Implementing the attention module of Bahdanau et al. 2015 where
    # score(h_j, s_(i-1)) = v . tanh(W_1 h_j + W_2 s_(i-1))
    def __init__(self, encoder_hidden_state_dim, decoder_hidden_state_dim, internal_dim=None):
        super(AdditiveAttention, self).__init__()
        self.encoder_hidden_state_dim = encoder_hidden_state_dim
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        if internal_dim is None:
            internal_dim = int((encoder_hidden_state_dim + decoder_hidden_state_dim) / 2)
        self.internal_dim = internal_dim
        self.w1 = nn.Linear(encoder_hidden_state_dim, internal_dim, bias=False)
        self.w2 = nn.Linear(decoder_hidden_state_dim, internal_dim, bias=False)
        self.v = nn.Linear(internal_dim, 1, bias=False)

    def score(self, encoder_state, decoder_state):
        # encoder_state is of shape (batch, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        # return value should be of shape (batch, 1)
        return self.v(torch.tanh(self.w1(encoder_state) + self.w2(decoder_state)))

    def forward(self, encoder_states, decoder_state):
        # encoder_states is of shape (batch, num_enc_states, enc_dim)
        # decoder_state is of shape (batch, dec_dim)
        score_vec = torch.cat([self.score(encoder_states[:, i], decoder_state) for i in range(encoder_states.shape[1])],
                              dim=1)
        # score_vec is of shape (batch, num_enc_states)

        attention_probs = torch.unsqueeze(F.softmax(score_vec, dim=1), dim=2)
        # attention_probs is of shape (batch, num_enc_states, 1)

        final_context_vec = torch.sum(attention_probs * encoder_states, dim=1)
        # final_context_vec is of shape (batch, enc_dim)

        return final_context_vec, attention_probs

if __name__ == "__main__":
    # model = RasterizedMapUNet(model_arch="resnet18", input_image_shape=(15, 224, 224), output_channel=4)
    t = torch.randn(2, 15, 224, 224)
    centers = torch.randint(low=10, high=214, size=(10, 2)).float()
    yaws = torch.zeros(size=(10, 1))
    extents = torch.ones(10, 2) * 2
    from tbsim.utils.geometry_utils import get_box_world_coords, get_square_upright_box
    boxes = get_box_world_coords(pos=centers, yaw=yaws, extent=extents)
    boxes_aligned = torch.flatten(boxes[:, [0, 2], :], start_dim=1)
    box_aligned = get_square_upright_box(centers, 2)
    from IPython import embed
    embed()
    boxes_indices = torch.zeros(10, 1)
    boxes_indices[5:] = 1
    boxes_indexed = torch.cat((boxes_indices, boxes_aligned), dim=1)

    model = RasterizeROIEncoder(
        model_arch="resnet50", input_image_shape=(15, 224, 224))
    output = model(t, boxes_indexed)
    for f in output:
        print(f.shape)

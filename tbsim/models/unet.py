import torch
import torch.nn as nn
import torch.nn.functional as F
# adapted from predictionnet by Nvidia
from tbsim.models.layers import (
    Activation,
    Normalization2D,
    Upsample2D)
class UResBlock(nn.Module):
    """UNet ResBlock.

    Uses dense 3x3 convs only, single resize *after* skip,
    and other graph tweaks specific for UNet.
    """

    def __init__(self, in_channels, out_channels, stride, upsample=None,
                 norm='bn', act='relu', out_act='relu'):
        """Construct UNet ResBlock."""
        super().__init__()
        self.act = Activation(out_act)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            Normalization2D(norm, in_channels),
            Activation(act),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            Activation(act),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
        self.resize = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if stride > 1:
            if upsample is not None:
                self.resize = Upsample2D(
                    upsample,
                    in_channels,
                    out_channels,
                    stride * 2 - 1,
                    stride=stride,
                    padding=stride // 2,
                    output_padding=0
                )
            else:
                self.resize = nn.Conv2d(
                    in_channels,
                    out_channels,
                    stride * 2 - 1,
                    stride=stride,
                    padding=stride // 2
                )

    def forward(self, x):
        """Forward pass."""
        x = self.block(x) + x
        return self.act(self.resize(x))

class UNet(nn.Module):
    """UNet model."""

    def __init__(self, in_frames, out_frames, in_channels, out_channels, channels, strides,
                 decoder_strides=None, skip_type='add', upsample='interp', norm='bn',
                 activation='relu', last_layer_prior=0.0001, last_layer_mvec_bias=0.0,
                 last_layer_vel_bias=0.0, enc_out_dropout=0.0, dec_in_dropout=0.0,
                 trajectory_candidates=0, enable_nll=False, desired_size=None, **kwargs):
        """Initialize the model.

        Args:
            in_frames: number of input frames.
            out_frames: number of output frames.
            in_channels: list of [static input channels, dynamic input channels].
            out_channels: list of [output 0 channels, ..., output N channels] for raster outputs.
            channels: number of channels in convolutional layers specified with a list.
            strides: encoder's convolutional strides specified with a list.
            decoder_strides: decoder's convolutional strides specified with a list.
                channels, strides, and decoder strides are specified with the same sized list.
            skip_type: connection between corresponding encoder/decoder blocks.
                Can be add (default), none.
            upsample: upsampling type in decoder. Passed to mode parameter of layers.Upsample2D.
            norm: normalization type. Can be batch norm (bn) or none.
            activation: activation function for all layers. See layers.Activation.
            last_layer_prior: last layer bias initialization prior to work properly with
                focal loss.
            last_layer_mvec_bias: last layer motion vector output initial bias.
            last_layer_vel_bias: last layer velocity output initial bias.
            enc_out_dropout: encoder output dropout.
            dec_in_dropout: decoder input dropout.
            trajectory_candidates: number of trajectories to regress directly.
            enable_nll: regress output for nll loss if true
            desired_size: desired size for the input
        """
        super(UNet, self).__init__()
        self._in_frames  = in_frames # noqa: disable=E221
        self._out_frames = out_frames
        self._out_channels = out_channels
        if isinstance(out_channels, int):
            self._out_channels = [out_channels]
        self._skip_type = self._get_str_arg(skip_type, 'skip_type', ['add', 'none'])
        self._norm = self._get_str_arg(norm, 'norm', ['bn', 'none'])
        self._act = self._get_str_arg(activation, 'activation', ['lrelu', 'relu', 'elu',
                                                                 'smooth_elu', 'sigmoid',
                                                                 'tanh', 'none'])
        self._channels = list(channels)
        self._strides  = list(strides) # noqa: disable=E221
        if decoder_strides is None:
            # symmetrical case, use enc. strides
            self._dec_strides = list(reversed(self._strides))
        else:
            self._dec_strides = list(decoder_strides)
        if len(self._channels) != len(self._strides):
            raise ValueError('Number of channels {} must be equal to number of '
                             'strides {}'.format(len(self._channels), len(self._strides)))
        self._num_resnet_blocks = len(self._channels) - 1
        static_in_channels, dynamic_in_channels = in_channels
        self._in_channels = dynamic_in_channels * in_frames + static_in_channels
        # Encoder.
        self._e0 = nn.Sequential(
            nn.Conv2d(self._in_channels, self._channels[0], kernel_size=3,
                      stride=self._strides[0], padding=1),
            Activation(self._act)
        )
        for i in range(1, self._num_resnet_blocks + 1):
            block = UResBlock(
                self._channels[i - 1],
                self._channels[i],
                stride=self._strides[i],
                upsample=None,
                norm=self._norm,
                act=self._act,
                out_act=self._act
            )
            setattr(self, '_e{}'.format(i), block)
        self._enc_drop = nn.Dropout2d(p=enc_out_dropout)
        self._dec_drop = nn.Dropout2d(p=dec_in_dropout)
        output_per_frame = 2  # acce, yaw rate
        if enable_nll:
            output_per_frame += 3  # log_stdx, log_stdy, rho
        total_out_channels = (
            sum(self._out_channels) * out_frames +  # BEV Image outputs
            trajectory_candidates * out_frames * output_per_frame  # Trajectory outputs
        )
        # Decoder.
        min_last_dec_channel = 64
        for i in range(1, self._num_resnet_blocks + 1):
            dec_in_channels_i = self._channels[i]
            dec_out_channels_i = self._channels[i - 1]
            if i == 1:
                # Last decoder channel doesn't need to match encoder because there's no skip
                # connection after
                dec_out_channels_i = max(dec_out_channels_i, min_last_dec_channel)
            block = UResBlock(
                dec_in_channels_i,
                dec_out_channels_i,
                stride=self._dec_strides[self._num_resnet_blocks - i],
                upsample=upsample,
                norm=self._norm,
                act=self._act,
                out_act=None
            )
            setattr(self, '_d{}'.format(i), block)
        self._dec_act = Activation(self._act)
        self._trajectory_candidates = trajectory_candidates
        # Output layers.
        # These are combined into one for efficiency;
        # each output head just slices the combined result
        output_layer = nn.Sequential(
            Upsample2D(upsample, max(min_last_dec_channel, self._channels[0]),
                       total_out_channels,
                       kernel_size=3, stride=self._dec_strides[-1],
                       padding=1, output_padding=0)
        )
        # Focal loss init for last layer occupancy channel
        last_bias_layer = [m for m in output_layer.modules() if hasattr(m, 'bias')][-1]
        with torch.no_grad():
            # most channels are zero-initialized
            last_bias_layer.bias.data[:] = 0
            # channels corresponding to occupancy get focal loss initialization
            occ_bias = -torch.log((torch.ones(1) - last_layer_prior) / last_layer_prior)
            last_bias_layer.bias.data[:out_frames] = occ_bias
            # Initialize biases for mvecs and velocity outputs.
            # If there are 2 or more outputs, mvecs will be in the last 2 channels.
            if len(out_channels) > 1:
                last_bias_layer.bias.data[-2 * out_frames:] = last_layer_mvec_bias
            # If there are 3 or more outputs, velocity will be in the second group of channels.
            if len(out_channels) > 2:
                last_bias_layer.bias.data[1*out_frames:3 * out_frames] = last_layer_vel_bias
        # Final outputs will be slices of _d0_core
        setattr(self, '_d0_core', output_layer)
        self.desired_size = desired_size

    @property
    def out_channels(self):
        """Returns output channels configuration."""
        return self._out_channels

    @staticmethod
    def _get_str_arg(val, name, allowed_values):
        if val not in allowed_values:
            raise ValueError('{} has invalid value {}. Supported values: {}'.format(
                             name, val, ', '.join(allowed_values)))
        if val == 'none':
            val = None
        return val

    def _skip_op(self, enc, dec):
        if self._skip_type == 'add':
            res = enc + dec
        elif self._skip_type is None:
            res = dec
        else:
            raise ValueError('Skip type {} is not supported.'.format(self._skip_type))
        return res

    def _get_skip_key_for_tensor(self, x):
        """Get a key for determining valid skip connections.

        Args:
            x: torch Tensor of NCHW data to be used in skip connection.
        Returns a key k(x) such that, if k(x) == k(y), then
        self._skip_op(x, y) is a valid skip connection.
        This is useful when the encoder / decoder are not using matching
        block counts / filter counts / strides, and symmetric UNet
        skip connections based only on the block index won't work.
        Example usage for a simple case:
        skips = {}
        # during encoder(x)
        skips[self._get_skip_key_for_tensor(x)] = x
        # ...
        # during decoder(y)
        y_key = self._get_skip_key_for_tensor(y)
        if y_key in skips:
            y = self._skip_op(skips.pop(y_key), y) # pop() to prevent reuse
        """
        assert x.ndim == 4, f'Invalid {x.shape}'
        # N is assumed to always match,
        # and H is assumed to match if W does
        # (also - broadcasting between H / TH is allowed),
        # so create a key just based on C and W.
        return (int(x.size(1)), int(x.size(-1)))

    def _apply_encoder(self, x, s):
        """Run the encoder tower on a batch of inputs.

        Args:
          x: N[CT]HW tensor, representing N episodes of T timesteps of
            C channels at HW spatial locations (aka dynamic context).
          s: NCHW tensor, representing N episodes of C channels
            at HW spatial locations (aka static context).
        Returns a tuple (
          a N[CT]HW tensor,
            containing final output features from the encoder,
          a dictionary of {size: NCHW tensor},
            containing intermediate features from each encoder block;
            these are only for the first timestep, and can therefore
            be safely used in decoder "skip connections" for all timesteps
            without leaking any information backwards in time.
        )
        """
        # Check sizes are reasonable
        assert (
            x.ndim == 4
        ), f'Invalid x {x.size()}, should be N[CT]HW'
        x = torch.cat([s, x], 1)
        e_in = self._e0(x)
        # Run all encoder blocks
        skip_out_dict = {}
        for i in range(self._num_resnet_blocks):
            block = getattr(self, '_e{}'.format(i + 1))
            e_out = block(e_in)
            e_in  = e_out # noqa: disable=E221
            skip_out_key = self._get_skip_key_for_tensor(e_out)
            skip_out_dict[skip_out_key] = skip_out_dict.get(skip_out_key, []) + [e_out]
        # Apply dropout on encoder output
        if self._enc_drop.p > 0:
            e_out = self._enc_drop(e_out)
        return e_out, skip_out_dict

    def _apply_decoder(self, x, skip_connections):
        """Run the decoder tower on a batch of inputs.

        Args:
          x: N[CT]HW tensor, representing N episodes of T timesteps of
            C channels at HW spatial locations (aka dynamic context).
          skip_connections: a dictionary of {size: NCHW tensor},
            representing output context features from each encoder stage.
        Returns an N[CT]HW tensor,
          containing decoder output. Output "heads" should
          be sliced from the channels of this tensor.
        """
        d_in = x
        # Apply dropout on decoder input
        if self._dec_drop.p > 0:
            d_in = self._dec_drop(d_in)
        # Run all decoder blocks, applying skip connections
        dec_skip_keys = []
        for i in range(self._num_resnet_blocks):
            # Apply skip connection
            skip_key = self._get_skip_key_for_tensor(d_in)
            dec_skip_keys.append(skip_key)
            if skip_key in skip_connections and len(skip_connections[skip_key]) > 0:
                skip = skip_connections[skip_key].pop()
                # do not add the same thing to itself!
                if d_in is not skip:
                    d_in = self._dec_act(self._skip_op(skip, d_in))
            else:
                print('failed skip for', skip_key, 'in', skip_connections.keys())
            # Apply block
            block = getattr(self, '_d{}'.format(self._num_resnet_blocks - i))
            d_out = block(d_in)
            d_in  = d_out # noqa: disable=E221
        # Apply shared decoder layer
        d_out = self._d0_core(self._dec_act(d_out))
        return d_out

    def is_recurrent(self):
        """Check if model is recurrent."""
        return False

    def forward(self, x):
        """Forward pass."""

        # The input contains separate static / dynamic tensors
        assert len(x) == 2, f'Invalid x of type {type(x)} length {len(x)}'
        s, x = x  # Separate static and dynamic parts
        assert (
            s.shape[-2:] == x.shape[-2:]
        ), f'Invalid static / dynamic shapes: ({s.shape}, {x.shape})'
        # In general, 5D tensors are consumed during training, while 4D - during inference.
        if x.dim() not in [4, 5]:
            raise ValueError(f'Expected 4D(N[CT]HW) or 5D(NCTHW) tensor but got {x.dim()}')
        if self.desired_size is not None:
            assert s.shape[-2]<=self.desired_size[0] and s.shape[-1]<=self.desired_size[1]
            w,h = s.shape[-2:]
            pw,ph = self.desired_size[0]-w,self.desired_size[1]-h
            pad_w = torch.zeros([*s.shape[:-2],pw,h],dtype=s.dtype,device=s.device)
            pad_h = torch.zeros([*s.shape[:-2],self.desired_size[0],ph],dtype=s.dtype,device=s.device)
            s = torch.cat((torch.cat((s,pad_w),-2),pad_h),-1)

            pad_w = torch.zeros([*x.shape[:-2],pw,h],dtype=x.dtype,device=x.device)
            pad_h = torch.zeros([*x.shape[:-2],self.desired_size[0],ph],dtype=x.dtype,device=x.device)
            x = torch.cat((torch.cat((x,pad_w),-2),pad_h),-1)


        # Save input H & W resolution

        dim_h_input, dim_w_input = x.size()[-2:]
        dim_n = int(x.size(0))
        input_is_5d = x.dim() == 5
        if input_is_5d:
            # Transform from NCTHW to N[CT]HW.
            s = s.view(dim_n, -1, dim_h_input, dim_w_input)
            x = x.view(dim_n, -1, dim_h_input, dim_w_input)

        # At this point, everything is 4D N[CT]HW
        # (regardless of training or export or whatever);
        # we'll convert back to NCTHW at the end if input_is_5d is True
        assert (
            s.size(1) + x.size(1) == self._in_channels
        ), f'Channel counts in input {s.size(), x.size()} do not match model'

        # Apply encoder / decoder
        enc_past, enc_past_skip = self._apply_encoder(x, s=s)
        res = self._apply_decoder(enc_past, enc_past_skip)

        # During export - don't do any resizing or slicing;
        # those steps will be handled by the DW inference code
        if torch.onnx.is_in_onnx_export():
            return res

        # During training, we upsample and slice as needed
        res = F.interpolate(
            res,
            size=(dim_h_input, dim_w_input),
            mode='bilinear',
            align_corners=False
        )

        # Slice outputs for each output head
        res_sliced = []
        slice_start = 0
        dim_t_out = self._out_frames
        dim_h_out, dim_w_out = (int(d) for d in res.size()[-2:])
        for dim_c_i in self._out_channels:
            slice_end = slice_start + dim_c_i * dim_t_out
            res_sliced_i = res[:, slice_start:slice_end]
            slice_start = slice_end
            # Slice is, again, N[CT]HW
            assert res_sliced_i.size() == (dim_n, dim_c_i * dim_t_out, dim_h_out, dim_w_out)
            if input_is_5d:
                # But, we convert the slice back to NCTHW (5d) if needed
                res_sliced_i = res_sliced_i.view(dim_n, dim_c_i, dim_t_out, dim_h_out, dim_w_out)
            res_sliced.append(res_sliced_i)

        # Optional trajectory regression output
        if self._trajectory_candidates > 0:
            # fill in Nones for any other outputs
            res_sliced.append(res[:, slice_end:])
        return tuple(res_sliced)


class DirectRegressionPostProcessor(torch.nn.Module):
    """Direct-regression trajectory extractor.

    This provides a differentiable module for converting a dense DNN output
    tensor into a trajectories, given initial positions.
    """

    def __init__(
        self,
        num_in_frames,
        image_height_px,
        image_width_px,
        sampling_time,
        trajectory_candidates,
        dyn
    ):
        """Initialize the post processor.

        Args:
          num_in_frames: the value of in_frames used for the predictions.
          image_height_px: the vertical spatial resolution of tensors to expect
            in postprocessor forward() - i.e. output resolution of prediction DNN.
          image_width_px: the corresponding horizontal spatial resolution.
          frame_duration_s: the time delta (seconds) between successive input / output frames
          trajectory_candidates: number of trajectory candidates to extract.
        """

        super().__init__()
        self._num_in_frames = num_in_frames
        self._image_height_px = image_height_px
        self._image_width_px = image_width_px
        assert (
            self._image_height_px == self._image_width_px
        ), f"Assumed square pixels, but {self._image_height_px} != {self._image_width_px}"

        self._dt_s = sampling_time
        assert self._dt_s > 0, f"Invalid frame duration (s): {self._dt_s}"
        self._dyn = dyn
        self._trajectory_candidates = trajectory_candidates

    def forward(self, obj_trajectory, query_pts, curr_state, actual_size=None):
        """Convert a predicted trajectory tensor and initial positions into a list of FrameTrajectories.

        Args:
          obj_trajectory: Torch tensor of dimensions [N, C, H, W] representing
            predicted trajectories at each location. For K candidates, C = K * (1 + T * 2),
            corresponding to 1 overall confidence score and T (ax, ay) acceleration pairs.
          query_pts: location of the agents on the image [N,Na,2]

        Returns a length-N list of FrameTrajectories for each episode in the batch.
        """
        bs,Na = query_pts.shape[:2]
        if actual_size is None:
            pos_xy_rel = query_pts.unsqueeze(1)/(obj_trajectory.shape[-1]-1)*2-1
        else:
            pos_xy_rel = query_pts.unsqueeze(1)/(actual_size-1)*2-1
        
        pred_per_agent = (
            torch.nn.functional.grid_sample(
                obj_trajectory,
                pos_xy_rel,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(2)
            .transpose(1,2)
        )
        
        input_pred = pred_per_agent.reshape(
             bs*Na, self._trajectory_candidates, -1, 2
        )  # AxTxCandidatesx2
        traj,pos,yaw = self._dyn.forward_dynamics(curr_state.reshape(bs*Na,1,-1).repeat_interleave(self._trajectory_candidates,1),input_pred,self._dt_s)
        
        return traj,pos,yaw,input_pred


def main():
    num_in_frames = 10
    num_out_frames = 10
    model = UNet(in_frames=num_in_frames, out_frames=num_out_frames, in_channels=[2,3], 
                 out_channels=[1], channels=[32, 64, 128, 128, 256, 256, 256], 
                 strides=[2, 2, 2, 2, 2, 2, 2], decoder_strides=[2, 2, 2, 2, 2, 1, 1],
                 skip_type='add', upsample='interp', norm='bn',
                 activation='relu', last_layer_prior=0.0001, last_layer_mvec_bias=0.0,
                 last_layer_vel_bias=0.0, enc_out_dropout=0.0, dec_in_dropout=0.0,
                 trajectory_candidates=2, enable_nll=False)

    static_input = torch.ones([10,2,255,255])
    dynamic_input = torch.zeros([10,3,10,255,255])
    out = model((static_input,dynamic_input))
    query_pts = torch.rand([10,5,2])*255
    from tbsim.dynamics import Unicycle
    dyn = Unicycle("vehicle", vbound=[-1, 40.0])
    curr_state=torch.randn([10,5,4])
    
    postprocessor = DirectRegressionPostProcessor(num_in_frames,
                                                    255,
                                                    255,
                                                    0.1,
                                                    2,
                                                    dyn,
                                                    )
    pred_logits, pred_traj = out
    traj = postprocessor(pred_traj,query_pts,curr_state)
    


if __name__=="__main__":
    main()
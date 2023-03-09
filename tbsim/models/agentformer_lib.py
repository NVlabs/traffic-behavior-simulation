"""
Modified version of PyTorch Transformer module for the implementation of Agent-Aware Attention (L290-L308)
"""

from typing import Callable, Dict, Final, List, Optional, Set, Tuple, Union
import warnings
import math
import copy

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import *
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.overrides import has_torch_function, handle_torch_function
from torchvision import models

def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist_dlow'].kl(data['p_z_dist_infer']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def diversity_loss(data, cfg):
    loss_unweighted = 0
    fut_motions = data['infer_dec_motion'].view(*data['infer_dec_motion'].shape[:2], -1)
    for motion in fut_motions:
        dist = F.pdist(motion, 2) ** 2
        loss_unweighted += (-dist / cfg['d_scale']).exp().mean()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def recon_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted



# """ DLow (Diversifying Latent Flows)"""
# class DLow(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.device = torch.device('cpu')
#         self.cfg = cfg
#         self.nk = nk = cfg.sample_k
#         self.nz = nz = cfg.nz
#         self.share_eps = cfg.get('share_eps', True)
#         self.train_w_mean = cfg.get('train_w_mean', False)
#         self.loss_cfg = self.cfg.loss_cfg
#         self.loss_names = list(self.loss_cfg.keys())

#         pred_cfg = Config(cfg.pred_cfg, cfg.train_tag, tmp=False, create_dirs=False)
#         pred_model = model_lib.model_dict[pred_cfg.model_id](pred_cfg)
#         self.pred_model_dim = pred_cfg.tf_model_dim
#         if cfg.pred_epoch > 0:
#             cp_path = pred_cfg.model_path % cfg.pred_epoch
#             print('loading model from checkpoint: %s' % cp_path)
#             model_cp = torch.load(cp_path, map_location='cpu')
#             pred_model.load_state_dict(model_cp['model_dict'])
#         pred_model.eval()
#         self.pred_model = [pred_model]

#         # Dlow's Q net
#         self.qnet_mlp = cfg.get('qnet_mlp', [512, 256])
#         self.q_mlp = MLP(self.pred_model_dim, self.qnet_mlp)
#         self.q_A = nn.Linear(self.q_mlp.out_dim, nk * nz)
#         self.q_b = nn.Linear(self.q_mlp.out_dim, nk * nz)
        
#     def set_device(self, device):
#         self.device = device
#         self.to(device)
#         self.pred_model[0].set_device(device)

#     def set_data(self, data):
#         self.pred_model[0].set_data(data)
#         self.data = self.pred_model[0].data

#     def main(self, mean=False, need_weights=False):
#         pred_model = self.pred_model[0]
#         if hasattr(pred_model, 'use_map') and pred_model.use_map:
#             self.data['map_enc'] = pred_model.map_encoder(self.data['agent_maps'])
#         pred_model.context_encoder(self.data)

#         if not mean:
#             if self.share_eps:
#                 eps = torch.randn([1, self.nz]).to(self.device)
#                 eps = eps.repeat((self.data['agent_num'] * self.nk, 1))
#             else:
#                 eps = torch.randn([self.data['agent_num'], self.nz]).to(self.device)
#                 eps = eps.repeat_interleave(self.nk, dim=0)

#         qnet_h = self.q_mlp(self.data['agent_context'])
#         A = self.q_A(qnet_h).view(-1, self.nz)
#         b = self.q_b(qnet_h).view(-1, self.nz)

#         z = b if mean else A*eps + b
#         logvar = (A ** 2 + 1e-8).log()
#         self.data['q_z_dist_dlow'] = Normal(mu=b, logvar=logvar)

#         pred_model.future_decoder(self.data, mode='infer', sample_num=self.nk, autoregress=True, z=z, need_weights=need_weights)
#         return self.data
    
#     def forward(self):
#         return self.main(mean=self.train_w_mean)

#     def inference(self, mode, sample_num, need_weights=False):
#         self.main(mean=True, need_weights=need_weights)
#         res = self.data[f'infer_dec_motion']
#         if mode == 'recon':
#             res = res[:, 0]
#         return res, self.data

#     def compute_loss(self):
#         total_loss = 0
#         loss_dict = {}
#         loss_unweighted_dict = {}
#         for loss_name in self.loss_names:
#             loss, loss_unweighted = loss_func[loss_name](self.data, self.loss_cfg[loss_name])
#             total_loss += loss
#             loss_dict[loss_name] = loss.item()
#             loss_unweighted_dict[loss_name] = loss_unweighted.item()
#         return total_loss, loss_dict, loss_unweighted_dict

#     def step_annealer(self):
#         pass

def agent_aware_attention(query: Tensor,
                          key: Tensor,
                          value: Tensor,
                          embed_dim_to_check: int,
                          num_heads: int,
                          in_proj_weight: Tensor,
                          in_proj_bias: Tensor,
                          bias_k: Optional[Tensor],
                          bias_v: Optional[Tensor],
                          add_zero_attn: bool,
                          dropout_p: float,
                          out_proj_weight: Tensor,
                          out_proj_bias: Tensor,
                          training: bool = True,
                          key_padding_mask: Optional[Tensor] = None,
                          need_weights: bool = True,
                          attn_mask: Optional[Tensor] = None,
                          use_separate_proj_weight: bool = False,
                          q_proj_weight: Optional[Tensor] = None,
                          k_proj_weight: Optional[Tensor] = None,
                          v_proj_weight: Optional[Tensor] = None,
                          static_k: Optional[Tensor] = None,
                          static_v: Optional[Tensor] = None,
                          gaussian_kernel = True,
                          num_agent = 1,
                          in_proj_weight_self = None,
                          in_proj_bias_self = None
                          ) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bs, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # # replace nan with 0, so that we can use to check if it is a self-attention vs cross-attention
    # query_no_nan = torch.nan_to_num(query.clone())
    # key_no_nan   = torch.nan_to_num(key.clone())
    # value_no_nan = torch.nan_to_num(value.clone())

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):     # PN x B X feat
        # if torch.equal(query_no_nan, key_no_nan) and torch.equal(key_no_nan, value_no_nan):     # PN x B X feat
            
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)  # PN x B x feat
            if in_proj_weight_self is not None:
                q_self, k_self = linear(query, in_proj_weight_self, in_proj_bias_self).chunk(2, dim=-1)
        
        # elif torch.equal(key_no_nan, value_no_nan):
        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

            if in_proj_weight_self is not None:
                _w = in_proj_weight_self[:embed_dim, :]
                _b = in_proj_bias_self[:embed_dim]
                q_self = linear(query, _w, _b)

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _w = in_proj_weight_self[embed_dim:, :]
                _b = in_proj_bias_self[embed_dim:]
                k_self = linear(key, _w, _b)

        else:
            raise NotImplementedError

    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    # k, q, v has PN x B X feat, q maybe FN x B x feat

    # default gaussian_kernel = False
    if not gaussian_kernel:
        q = q * scaling       # remove scaling
        if in_proj_weight_self is not None:
            q_self = q_self * scaling       # remove scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bs * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bs, 1)])
            v = torch.cat([v, bias_v.repeat(1, bs, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bs * num_heads, head_dim).transpose(0, 1)         # BH x PN x feat
    if k is not None:
        k = k.contiguous().view(-1, bs * num_heads, head_dim).transpose(0, 1)          # BH x PN x feat
    if v is not None:
        v = v.contiguous().view(-1, bs * num_heads, head_dim).transpose(0, 1)
    if in_proj_weight_self is not None:
        q_self = q_self.contiguous().view(tgt_len, bs * num_heads, head_dim).transpose(0, 1)
        k_self = k_self.contiguous().view(-1, bs * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bs * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bs * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bs
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    if gaussian_kernel:
        qk = torch.bmm(q, k.transpose(1, 2))
        q_n = q.pow(2).sum(dim=-1).unsqueeze(-1)
        k_n = k.pow(2).sum(dim=-1).unsqueeze(1)
        qk_dist = q_n + k_n - 2 * qk
        attn_output_weights = qk_dist * scaling * 0.5
    else:
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))   # BH x PN x PN, or BH x FN x PN
        # attention weights contain random numbers for the timestamps without data

    assert list(attn_output_weights.size()) == [bs * num_heads, tgt_len, src_len]

    if in_proj_weight_self is not None:
        """
        ==================================
            Agent-Aware Attention
        ==================================
        """
        attn_output_weights_inter = attn_output_weights                         # BH x PN x PN, or BH x FN x PN
        attn_output_weights_self = torch.bmm(q_self, k_self.transpose(1, 2))    # BH x PN x PN, or BH x FN x PN

        # using identity matrix here since the agents are not shuffled
        attn_weight_self_mask = torch.eye(num_agent).to(q.device)
        attn_weight_self_mask = attn_weight_self_mask.repeat([attn_output_weights.shape[1] // num_agent, attn_output_weights.shape[2] // num_agent]).unsqueeze(0)   
        # 1 x PN x PN
     
        attn_output_weights = attn_output_weights_inter * (1 - attn_weight_self_mask) + attn_output_weights_self * attn_weight_self_mask    # BH x PN x PN

        # masking the columns / rows
        if attn_mask is not None:                   # BH x PN x PN or BH x FN x PN
            if attn_mask.dtype == torch.bool:
                
                # assign -inf so that the columns of the invalid data will lead to 0 after softmax during attention
                # this is to disable the interaction between valid agents in rows and invalid agents in some columns
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))  # BH x PN x PN or BH x FN x PN

                # however, the rows with invalid data (with all -inf now) will lead to NaN after softmax 
                # because there is no single valid data for that row
                # as a result, it will lead to NaN in backward during softmax
                # we need to assign some dummy numbers to it, this process is needed for training
                # but this does not affect the results in forward pass since these rows of features are not used
                attn_output_weights.masked_fill_(attn_mask[:, :, [0]], 0.0)  
            else:
                attn_output_weights += attn_mask                            # BH x PN x PN or BH x FN x PN

        attn_output_weights = softmax(attn_output_weights, dim=-1)          # BH x PN x PN or BH x FN x PN
        # the output attn_output_weights will have 0.17 for the rows with invalid data
        # because the entire row prior to softmax is 0, so it is averaged over the row

        # to suppress the random number in the row with invalida data
        # we mask again with 0s, however in-place operation not supported for backward
        # attn_output_weights.masked_fill_(attn_mask[[0], :, [0]].unsqueeze(-1), 0.0)  
    else:
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(bs, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_output_weights = attn_output_weights.view(bs * num_heads, tgt_len, src_len)
        attn_output_weights = softmax(
            attn_output_weights, dim=-1)

    # attn_output_weights is row-wise, i.e., the agent at a timestamp without valid data (NaN) has some random numbers
    # for the columns, the agent at a timestamp without valid data is 0
    # add torch.nan_to_num to convert NaN to a large number for the agent value without valid data
    # but when the 0 in attn_output_weights col * the large number in v, it will result in 0
    # in other words, we do not attend to the agent timestamp with NaN data
    # the output might have some invalid rows (rows with random numbers but do not affect results)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)  # BH x PN x PN, or BH x FN x PN
    attn_output = torch.bmm(attn_output_weights, torch.nan_to_num(v, nan=1e+10))        # BH x PN x feat, or BH x FN x feat

    # to maintain elegancy, we mask those invalid rows with random numbers as 0s
    # but not masking will not affect results
    # final_mask = attn_mask[:, :, [0]]     # 1 x PN x 1
    # attn_output = attn_output.masked_fill_(final_mask, 0.0)     # BH x PN x feat

    # convert to output shape
    assert list(attn_output.size()) == [bs * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, embed_dim)     # PN x B x feat
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)                       # PN x B x feat

    # average attention weights over heads
    if need_weights:
        attn_output_weights = attn_output_weights.view(bs, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class AgentAwareAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, cfg, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super().__init__()
        self.cfg = cfg
        self.gaussian_kernel = self.cfg.get('gaussian_kernel', False)
        self.sep_attn = self.cfg.get('sep_attn', True)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if self.sep_attn:
            self.in_proj_weight_self = Parameter(torch.empty(2 * embed_dim, embed_dim))
            self.in_proj_bias_self = Parameter(torch.empty(2 * embed_dim))
        else:
            self.in_proj_weight_self = self.in_proj_bias_self = None

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

        if self.sep_attn:
            xavier_uniform_(self.in_proj_weight_self)
            constant_(self.in_proj_bias_self, 0.)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, num_agent=1):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return agent_aware_attention(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight, gaussian_kernel=self.gaussian_kernel,
                num_agent=num_agent,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self
                )
        else:
            return agent_aware_attention(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, gaussian_kernel=self.gaussian_kernel,
                num_agent=num_agent,
                in_proj_weight_self=self.in_proj_weight_self,
                in_proj_bias_self=self.in_proj_bias_self
                )


class AgentFormerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, num_agent=1) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, num_agent=num_agent)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class AgentFormerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, cfg, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.cfg = cfg
        self.self_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        self.multihead_attn = AgentAwareAttention(cfg, d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, num_agent = 1, need_weights = False) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask, num_agent=num_agent, need_weights=need_weights)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask, num_agent=num_agent, need_weights=need_weights)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, self_attn_weights, cross_attn_weights


class AgentFormerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, num_agent=1) -> torch.Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, num_agent=num_agent)

        if self.norm is not None:
            output = self.norm(output)

        return output


class AgentFormerDecoder(Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, num_agent=1, need_weights = False) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        self_attn_weights = [None] * len(self.layers)
        cross_attn_weights = [None] * len(self.layers)
        for i, mod in enumerate(self.layers):
            output, self_attn_weights[i], cross_attn_weights[i] = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask,
                         num_agent=num_agent, need_weights=need_weights)

        if self.norm is not None:
            output = self.norm(output)

        if need_weights:
            self_attn_weights = torch.stack(self_attn_weights).cpu().numpy()
            cross_attn_weights = torch.stack(cross_attn_weights).cpu().numpy()

        return output, {'self_attn_weights': self_attn_weights, 'cross_attn_weights': cross_attn_weights}


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class MapCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.convs = nn.ModuleList()
        map_channels = cfg.get('map_channels', 3)
        patch_size = cfg.get('patch_size', [100, 100])
        hdim = cfg.get('hdim', [32, 32])
        kernels = cfg.get('kernels', [3, 3])
        strides = cfg.get('strides', [3, 3])
        self.out_dim = out_dim = cfg.get('out_dim', 32)
        self.input_size = input_size = (map_channels, patch_size[0], patch_size[1])
        x_dummy = torch.randn(input_size).unsqueeze(0)

        for i, _ in enumerate(hdim):
            self.convs.append(nn.Conv2d(map_channels if i == 0 else hdim[i-1],
                                        hdim[i], kernels[i],
                                        stride=strides[i]))
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), out_dim)

    def forward(self, x):
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class MapEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_id = cfg.get('model_id', 'map_cnn')
        dropout = cfg.get('dropout', 0.0)
        self.normalize = cfg.get('normalize', True)
        self.dropout = nn.Dropout(dropout)
        if model_id == 'map_cnn':
            self.model = MapCNN(cfg)
            self.out_dim = self.model.out_dim
        elif 'resnet' in model_id:
            model_dict = {
                'resnet18': models.resnet18,
                'resnet34': models.resnet34,
                'resnet50': models.resnet50
            }
            self.out_dim = out_dim = cfg.get('out_dim', 32)
            self.model = model_dict[model_id](pretrained=False, norm_layer=nn.InstanceNorm2d)
            self.model.fc = nn.Linear(self.model.fc.in_features, out_dim)
        else:
            raise ValueError('unknown map encoder!')

    def forward(self, x):
        if self.normalize:
            x = x * 2. - 1.
        x = self.model(x)
        x = self.dropout(x)
        return x

def compute_motion_mse(data, cfg):
    diff = data['fut_motion_orig'] - data['train_dec_motion']
    # print(data['fut_motion_orig'])
    # print(data['train_dec_motion'])
    # zxc
    if cfg.get('mask', True):
        mask = data['fut_mask']
        diff *= mask.unsqueeze(2)
    loss_unweighted = diff.pow(2).sum() 
    if cfg.get('normalize', True):
        loss_unweighted /= diff.shape[0]
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_z_kld(data, cfg):
    loss_unweighted = data['q_z_dist'].kl(data['p_z_dist']).sum()
    if cfg.get('normalize', True):
        loss_unweighted /= data['batch_size']
    loss_unweighted = loss_unweighted.clamp_min_(cfg.min_clip)
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


def compute_sample_loss(data, cfg):
    diff = data['infer_dec_motion'] - data['fut_motion_orig'].unsqueeze(1)
    if cfg.get('mask', True):
        mask = data['fut_mask'].unsqueeze(1).unsqueeze(-1)
        diff *= mask
    dist = diff.pow(2).sum(dim=-1).sum(dim=-1)
    loss_unweighted = dist.min(dim=1)[0]
    if cfg.get('normalize', True):
        loss_unweighted = loss_unweighted.mean()
    else:
        loss_unweighted = loss_unweighted.sum()
    loss = loss_unweighted * cfg['weight']
    return loss, loss_unweighted


loss_func = {
    'mse': compute_motion_mse,
    'kld': compute_z_kld,
    'sample': compute_sample_loss
}
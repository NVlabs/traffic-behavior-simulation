from logging import raiseExceptions
import numpy as np

import torch
import math, copy
from typing import Dict
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tbsim.utils.tensor_utils as TensorUtils


def clones(module, n):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class FactorizedEncoderDecoder(nn.Module):
    """
    A encoder-decoder transformer model with Factorized encoder and decoder
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator, src2posfun):
        """
        Args:
            encoder: FactorizedEncoder
            decoder: FactorizedDecoder
            src_embed: source embedding network
            tgt_embed: target embedding network
            generator: network used to generate output from target
            src2posfun: extract positional info from the src
        """
        super(FactorizedEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.src2posfun = src2posfun

    def src2pos(self, src, dyn_type):
        "extract positional info from src for all datatypes, e.g., for vehicles, the first two dimensions are x and y"

        pos = torch.zeros([*src.shape[:-1], 2]).to(src.device)
        for dt, fun in self.src2posfun.items():
            pos += fun(src) * (dyn_type == dt).view([*(dyn_type.shape), 1, 1])

        return pos

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        tgt_mask_agent,
        dyn_type,
        map_emb=None,
    ):
        "Take in and process masked src and target sequences."
        src_pos = self.src2pos(src, dyn_type)
        "for decoders, we only use position at the last time step of the src"
        return self.decode(
            self.encode(src, src_mask, src_pos, map_emb),
            src_mask,
            tgt,
            tgt_mask,
            tgt_mask_agent,
            src_pos[:, :, -1:],
        )

    def encode(self, src, src_mask, src_pos, map_emb):
        return self.encoder(self.src_embed(src), src_mask, src_pos, map_emb)

    def decode(self, memory, src_mask, tgt, tgt_mask, tgt_mask_agent, pos):

        return self.decoder(
            self.tgt_embed(tgt),
            memory,
            src_mask,
            tgt_mask,
            tgt_mask_agent,
            pos,
        )


class DynamicGenerator(nn.Module):
    "Incorporating dynamics to the generator to generate dynamically feasible output, not used yet"

    def __init__(self, d_model, dt, dyns, state2feature, feature2state):
        super(DynamicGenerator, self).__init__()
        self.dyns = dyns
        self.proj = dict()
        self.dt = dt
        self.state2feature = state2feature
        self.feature2state = feature2state
        for dyn in self.dyns:
            self.proj[dyn.type()] = nn.Linear(d_model, dyn.udim)

    def forward(self, x, tgt, type_index):
        Nagent = tgt.shape[0]
        tgt_next = [None] * Nagent
        for dyn in self.dyns:
            index = type_index[dyn.type()]
            state = self.feature2state[dyn.name](tgt[index])
            input = self.proj[dyn.type()](x)
            state_next = dyn.step(state, input, self.dt)
            x_next_raw = self.state2feature[dyn.name](state_next)
            for i in range(len(index)):
                tgt_next[index[i]] = x_next_raw[i]
        return torch.stack(tgt_next, dim=0)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, mask1=None):
        "Pass the input (and mask) through each layer in turn."
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(x, mask)
            else:
                if mask1 is None:
                    x = layer(x, mask)
                else:
                    x = layer(x, mask1)
        return self.norm(x)


class FactorizedEncoder(nn.Module):
    def __init__(self, temporal_enc, agent_enc, temporal_pe, XY_pe, N_layer=1):
        """
        Factorized encoder and agent axis
        Args:
            temporal_enc: encoder with attention over temporal axis
            agent_enc: encoder with attention over agent axis
            temporal_pe: positional encoding over time
            XY_pe: positional encoding over XY coordinates
        """
        super(FactorizedEncoder, self).__init__()
        self.N_layer = N_layer
        self.temporal_encs = clones(temporal_enc, N_layer)
        self.agent_encs = clones(agent_enc, N_layer)
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, src_mask, src_pos, map_emb):
        """Pass the input (and mask) through each layer in turn.
        Args:
            x:[B,Num_agent,T,d_model]
            src_mask:[B,Num_agent,T]
            src_pos:[B,Num_agent,T,2]
            map_emb: [B,Num_agent,1,map_emb_dim] output of the CNN ROI map encoder
        Returns:
            embedding of size [B,Num_agent,T,d_model]
        """

        if map_emb.ndim == 3:
            map_emb = map_emb.unsqueeze(2).repeat(1, 1, x.size(2), 1)
        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, src_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                    map_emb,
                ),
                dim=-1,
            )
            * src_mask.unsqueeze(-1)
        )
        for i in range(self.N_layer):
            x = self.agent_encs[i](x, src_mask)
            x = self.temporal_encs[i](x, src_mask)
        return x


class StaticEncoder(nn.Module):
    def __init__(self, agent_enc, XY_pe, N_layer=1):
        """
        Factorized encoder and agent axis
        Args:
            temporal_enc: encoder with attention over temporal axis
            agent_enc: encoder with attention over agent axis
            temporal_pe: positional encoding over time
            XY_pe: positional encoding over XY coordinates
        """
        super(StaticEncoder, self).__init__()
        self.N_layer = N_layer
        self.agent_encs = clones(agent_enc, N_layer)
        self.XY_pe = XY_pe

    def forward(self, x, src_mask, src_pos, map_emb=None):
        """Pass the input (and mask) through each layer in turn.
        Args:
            x:[B,Num_agent,T,d_model]
            src_mask:[B,Num_agent,T]
            src_pos:[B,Num_agent,T,2]
            map_emb: [B,Num_agent,1,map_emb_dim] output of the CNN ROI map encoder
        Returns:
            embedding of size [B,Num_agent,T,d_model]
        """
        inputs = [x, self.XY_pe(x, src_pos)]
        if map_emb is not None:
            inputs.append(map_emb)

        x = (
            torch.cat(
                (
                    inputs
                ),
                dim=-1,
            )
            * src_mask.unsqueeze(-1)
        )
        for i in range(self.N_layer):
            x = self.agent_encs[i](x, src_mask)
        return x


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "self attention followed by feedforward, residual and batch norm in between layers"
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        "cross attention to the embedding generated by the encoder"
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SummaryModel(nn.Module):
    """
    map the scene information to attributes that summarizes the scene
    """

    def __init__(self, encoder, decoder, src_embed, src2posfun):
        super(SummaryModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src2posfun = src2posfun

    def src2pos(self, src, dyn_type):
        "extract positional info from src for all datatypes, e.g., for vehicles, the first two dimensions are x and y"

        pos = torch.zeros([*src.shape[:-1], 2]).to(src.device)
        for dt, fun in self.src2posfun.items():
            pos += fun(src) * (dyn_type == dt).view([*(dyn_type.shape), 1, 1])

        return pos

    def forward(
        self,
        src,
        src_mask,
        dyn_type,
        map_emb,
    ):
        "Take in and process masked src and target sequences."
        src_pos = self.src2pos(src, dyn_type)
        return self.decode(
            self.encode(src, src_mask, src_pos, map_emb),
            src_mask,
        )

    def encode(self, src, src_mask, src_pos, map_emb):
        return self.encoder(self.src_embed(src), src_mask, src_pos, map_emb)

    def decode(self, memory, src_mask):
        return self.decoder(memory, src_mask)


class SummaryDecoder(nn.Module):
    """
    Map the encoded tensor to a description of the whole scene, e.g., the likelihood of certain modes
    """

    def __init__(
        self, temporal_attn, agent_attn, ff, emb_dim, output_dim, static=False
    ):
        super(SummaryDecoder, self).__init__()
        self.temporal_attn = temporal_attn
        self.agent_attn = agent_attn
        self.ff = ff
        self.output_dim = output_dim
        self.static = static
        self.MLP = nn.Sequential(nn.Linear(emb_dim, output_dim), nn.Sigmoid())

    def forward(self, x, mask):
        x = self.agent_attn(x, x, x, mask)
        x = self.ff(torch.max(x, dim=-3)[0]).unsqueeze(1)
        if not self.static:
            x = self.temporal_attn(x, x, x)
            x = torch.max(x, dim=-2)[0].squeeze(1)
        x = self.MLP(x)
        return x


class FactorizedDecoder(nn.Module):
    """
    Args:
        temporal_dec: decoder with attention over temporal axis
        agent_enc: decoder with attention over agent axis
        temporal_pe: positional encoding for time axis
        XY_pe: positional encoding for XY axis
    """

    def __init__(
        self,
        temporal_dec,
        agent_enc,
        temporal_enc,
        temporal_pe,
        XY_pe,
        N_layer_enc=1,
        N_layer_dec=1,
    ):
        super(FactorizedDecoder, self).__init__()
        self.temporal_dec = clones(temporal_dec, N_layer_dec)
        self.agent_enc = clones(agent_enc, N_layer_enc)
        self.temporal_enc = clones(temporal_enc, N_layer_enc)
        self.N_layer_enc = N_layer_enc
        self.N_layer_dec = N_layer_dec
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe

    def forward(self, x, memory, src_mask, tgt_mask, tgt_mask_agent, pos):
        """
        Pass the input (and mask) through each layer in turn.
        Args:
            x (torch.tensor)): [batch,Num_agent,T_tgt,d_model]
            memory (torch.tensor): [batch,Num_agent,T_src,d_model]
            src_mask (torch.tensor): [batch,Num_agent,T_src]
            tgt_mask (torch.tensor): [batch,Num_agent,T_tgt]
            tgt_mask_agent (torch.tensor): [batch,Num_agent,T_tgt]
            pos (torch.tensor): [batch,Num_agent,1,2]

        Returns:
            torch.tensor: [batch,Num_agent,T_tgt,d_model]
        """
        T = x.size(-2)
        tgt_pos = pos.repeat([1, 1, T, 1])

        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, tgt_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                ),
                dim=-1,
            )
            * tgt_mask_agent.unsqueeze(-1)
        )

        for i in range(self.N_layer_dec):
            x = self.temporal_dec[i](x, memory, src_mask, tgt_mask)
        prob = torch.ones(x.shape[0]).to(x.device)
        return x * tgt_mask_agent.unsqueeze(-1), prob


class MultimodalFactorizedDecoder(nn.Module):
    """
    Args:
        temporal_dec: decoder with attention over temporal axis
        agent_enc: decoder with attention over agent axis
        temporal_pe: positional encoding for time axis
        XY_pe: positional encoding for XY axis
    """

    def __init__(
        self,
        temporal_dec,
        agent_enc,
        temporal_enc,
        temporal_pe,
        XY_pe,
        M,
        summary_dec,
        N_layer_enc=1,
        N_layer_dec=1,
    ):
        super(MultimodalFactorizedDecoder, self).__init__()
        self.M = M
        self.temporal_dec = clones(temporal_dec, N_layer_dec)
        self.agent_enc = clones(agent_enc, N_layer_enc)
        self.temporal_enc = clones(temporal_enc, N_layer_enc)
        self.N_layer_enc = N_layer_enc
        self.N_layer_dec = N_layer_dec
        self.temporal_pe = temporal_pe
        self.XY_pe = XY_pe
        self.summary_dec = summary_dec

    def forward(self, x, memory, src_mask, tgt_mask, tgt_mask_agent, pos):
        """
        Pass the input (and mask) through each layer in turn.
        Args:
            x (torch.tensor)): [batch,Num_agent,T_tgt,d_model]
            memory (torch.tensor): [batch,Num_agent,T_src,d_model]
            src_mask (torch.tensor): [batch,Num_agent,T_src]
            tgt_mask (torch.tensor): [batch,Num_agent,T_tgt]
            tgt_mask_agent (torch.tensor): [batch,Num_agent,T_tgt]
            pos (torch.tensor): [batch,Num_agent,1,2]

        Returns:
            torch.tensor: [batch,Num_agent,T_tgt,d_model]
        """
        T = x.size(-2)
        tgt_pos = pos.repeat([1, 1, T, 1])

        x = (
            torch.cat(
                (
                    x,
                    self.XY_pe(x, tgt_pos),
                    self.temporal_pe(x).repeat(x.size(0), x.size(1), 1, 1),
                ),
                dim=-1,
            )
            * tgt_mask_agent.unsqueeze(-1)
        )

        # adding one-hot encoding of the modes
        modes_enc = (
            F.one_hot(torch.arange(0, self.M))
            .view(1, self.M, 1, 1, self.M)
            .repeat(x.size(0), 1, x.size(1), x.size(2), 1)
        ).to(x.device)

        x = torch.cat((x.unsqueeze(1).repeat(1, self.M, 1, 1, 1), modes_enc), dim=-1)

        memory_M = memory.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
        src_mask_M = src_mask.unsqueeze(1).repeat(1, self.M, 1, 1)
        tgt_mask_M = tgt_mask.unsqueeze(1).repeat(1, self.M, 1, 1, 1)
        tgt_mask_agent_M = tgt_mask_agent.unsqueeze(1).repeat(1, self.M, 1, 1)
        for i in range(self.N_layer_enc):
            x = self.agent_enc[i](x, tgt_mask_agent_M)
            x = self.temporal_enc[i](x, tgt_mask_M)
        for i in range(self.N_layer_dec):
            x = self.temporal_dec[i](
                x,
                memory_M,
                src_mask_M,
                tgt_mask_M,
            )

        prob = self.summary_dec(x, tgt_mask_agent_M).squeeze(-1)
        prob = F.softmax(prob, dim=-1)
        return x * tgt_mask_agent_M.unsqueeze(-1), prob


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "self attention followed by cross attention with the encoder output"

        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, pooling_dim=None):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.pooling_dim = pooling_dim
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if self.pooling_dim is None:
            pooling_dim = -2
        else:
            pooling_dim = self.pooling_dim

        if mask is not None:
            # Same mask applied to all h heads.
            if mask.ndim == query.ndim - 1:
                mask = mask.view([*mask.shape, 1, 1]).transpose(-1, pooling_dim - 1)
            elif mask.ndim == query.ndim:
                mask = mask.unsqueeze(-2).transpose(-2, pooling_dim - 1)
            else:
                raise Exception("mask dimension mismatch")

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [
            l(x).view(*x.shape[:-1], self.h, self.d_k)
            for l, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query.transpose(-2, pooling_dim - 1),
            key.transpose(-2, pooling_dim - 1),
            value.transpose(-2, pooling_dim - 1),
            mask,
            dropout=self.dropout,
        )

        x = x.transpose(-2, pooling_dim - 1).contiguous()
        x = x.view(*x.shape[:-2], self.h * self.d_k)

        # 3) "Concat" using a view and apply a final linear.
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, dim, dropout, max_len=5000, flipped=False):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.flipped = flipped

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        if self.flipped:
            position = -position.flip(dims=[0])
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe_shape = [1] * (x.ndim - 2) + list(x.shape[-2:-1]) + [self.dim]
        if self.flipped:
            return self.dropout(
                Variable(self.pe[:, -x.size(-2) :].view(pe_shape), requires_grad=False)
            )
        else:
            return self.dropout(
                Variable(self.pe[:, : x.size(-2)].view(pe_shape), requires_grad=False)
            )


class PositionalEncodingNd(nn.Module):
    "extension of the PE function, works for N dimensional position input"

    def __init__(self, dim, dropout, step_size=[1]):
        """
        step_size: scale of each dimension, pos/step_size = phase for the sinusoidal PE
        """
        super(PositionalEncodingNd, self).__init__()
        assert dim % 2 == 0
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim
        self.step_size = step_size
        self.D = len(step_size)
        self.pe = list()

        # Compute the positional encodings once in log space.
        self.div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    def forward(self, x, pos):
        rep_size = [1] * (x.ndim)
        rep_size[-1] = int(self.dim / 2)
        pe_shape = [*x.shape[:-1], self.dim]
        for i in range(self.D):
            pe = torch.zeros(pe_shape).to(x.device)

            pe[..., 0::2] = torch.sin(
                pos[..., i : i + 1].repeat(*rep_size)
                / self.step_size[i]
                * self.div_term.to(x.device)
            )
            pe[..., 1::2] = torch.sin(
                pos[..., i : i + 1].repeat(*rep_size)
                / self.step_size[i]
                * self.div_term.to(x.device)
            )
        return self.dropout(Variable(pe, requires_grad=False))


def make_transformer_model(
    src_dim,
    tgt_dim,
    out_dim,
    dyn_list,
    N_t=6,
    N_a=3,
    d_model=384,
    XY_pe_dim=64,
    temporal_pe_dim=64,
    map_emb_dim=128,
    d_ff=2048,
    head=8,
    dropout=0.1,
    step_size=[0.1, 0.1],
    N_layer_enc=1,
    N_layer_tgt_enc=1,
    N_layer_tgt_dec=1,
    M=1,
    use_GAN=False,
    GAN_static=True,
    N_layer_enc_discr=1,
):
    "first generate the building blocks, attn networks, encoders, decoders, PEs and Feedforward nets"
    c = copy.deepcopy
    temporal_attn = MultiHeadedAttention(head, d_model)
    agent_attn = MultiHeadedAttention(head, d_model, pooling_dim=-3)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    temporal_pe = PositionalEncoding(temporal_pe_dim, dropout)
    temporal_pe_flip = PositionalEncoding(temporal_pe_dim, dropout, flipped=True)
    XY_pe = PositionalEncodingNd(XY_pe_dim, dropout, step_size=step_size)
    temporal_enc = Encoder(EncoderLayer(d_model, c(temporal_attn), c(ff), dropout), N_t)
    agent_enc = Encoder(EncoderLayer(d_model, c(agent_attn), c(ff), dropout), N_a)

    src_emb = nn.Linear(src_dim, d_model - XY_pe_dim - temporal_pe_dim - map_emb_dim)
    if M == 1:
        tgt_emb = nn.Linear(tgt_dim, d_model - XY_pe_dim - temporal_pe_dim)
    else:
        tgt_emb = nn.Linear(tgt_dim, d_model - XY_pe_dim - temporal_pe_dim - M)
    generator = nn.Linear(d_model, out_dim)

    temporal_dec = Decoder(
        DecoderLayer(d_model, c(temporal_attn), c(temporal_attn), c(ff), dropout), N_t
    )
    "gather src2posfun from all agent types"
    src2posfun = {D.type(): D.state2pos for D in dyn_list}

    Factorized_Encoder = FactorizedEncoder(
        c(temporal_enc), c(agent_enc), temporal_pe_flip, XY_pe, N_layer_enc
    )
    if M == 1:
        Factorized_Decoder = FactorizedDecoder(
            c(temporal_dec),
            c(agent_enc),
            c(temporal_enc),
            temporal_pe,
            XY_pe,
            N_layer_tgt_enc,
            N_layer_tgt_dec,
        )
    else:
        mode_summary_dec = SummaryDecoder(
            c(temporal_attn), c(agent_attn), c(ff), d_model, 1
        )
        Factorized_Decoder = MultimodalFactorizedDecoder(
            temporal_dec,
            agent_enc,
            temporal_enc,
            temporal_pe,
            XY_pe,
            M,
            mode_summary_dec,
            N_layer_enc=1,
            N_layer_dec=1,
        )
    Factorized_Encoder = FactorizedEncoder(
        c(temporal_enc), c(agent_enc), temporal_pe_flip, XY_pe, N_layer_enc
    )
    if use_GAN:
        if GAN_static:
            Summary_Encoder = StaticEncoder(
                c(agent_enc),
                XY_pe,
                N_layer_enc_discr,
            )
            Summary_Decoder = SummaryDecoder(
                c(temporal_attn), c(agent_attn), c(ff), d_model, 1, static=True
            )
            static_src_emb = nn.Linear(src_dim, d_model - XY_pe_dim - map_emb_dim)
            Summary_Model = SummaryModel(
                Summary_Encoder,
                Summary_Decoder,
                c(static_src_emb),
                src2posfun,
            )
        else:
            Summary_Encoder = Summary_Encoder = FactorizedEncoder(
                c(temporal_enc),
                c(agent_enc),
                temporal_pe_flip,
                XY_pe,
                N_layer_enc_discr,
            )
            Summary_Decoder = SummaryDecoder(
                c(temporal_attn), c(agent_attn), c(ff), d_model, 1, static=True
            )
            Summary_Model = SummaryModel(
                Summary_Encoder,
                Summary_Decoder,
                c(src_emb),
                src2posfun,
            )

    else:
        Summary_Model = None
    "use a simple nn.Linear as the generator as our output is continuous"

    Transformer_Model = FactorizedEncoderDecoder(
        Factorized_Encoder,
        Factorized_Decoder,
        c(src_emb),
        c(tgt_emb),
        c(generator),
        src2posfun,
    )

    return Transformer_Model, Summary_Model


class SimpleTransformer(nn.Module):
    def __init__(
            self,
            src_dim,
            N_a=3,
            d_model=384,
            XY_pe_dim=64,
            d_ff=2048,
            head=8,
            dropout=0.1,
            step_size=[0.1, 0.1],
    ):
        super(SimpleTransformer, self).__init__()
        c = copy.deepcopy
        agent_attn = MultiHeadedAttention(head, d_model, pooling_dim=-3)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        XY_pe = PositionalEncodingNd(XY_pe_dim, dropout, step_size=step_size)
        self.agent_enc = StaticEncoder(EncoderLayer(d_model, c(agent_attn), c(ff), dropout), XY_pe, N_a)
        self.pre_emb = nn.Linear(src_dim, d_model - XY_pe_dim)
        self.post_emb = nn.Linear(d_model, src_dim)

    def forward(self, feats, avails, pos):
        x = self.pre_emb(feats)
        x = self.agent_enc(x, avails, pos)
        return self.post_emb(x)


class simplelinear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[64, 32]):
        super(simplelinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_layers = len(hidden_dim)
        self.fhidden = nn.ModuleList()

        for i in range(1, self.hidden_layers):
            self.fhidden.append(nn.Linear(hidden_dim[i - 1], hidden_dim[i]))

        self.f1 = nn.Linear(input_dim, hidden_dim[0])
        self.f2 = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        hidden = self.f1(x)
        for i in range(1, self.hidden_layers):
            hidden = self.fhidden[i - 1](F.relu(hidden))
        return self.f2(F.relu(hidden))

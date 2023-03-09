import torch
from collections import OrderedDict

from tbsim import dynamics
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import numpy as np
from torch import nn
from torch.nn import functional as F
from collections import defaultdict
from tbsim.utils.model_utils import AFMLP, Normal, Categorical, initialize_weights, rotation_2d_torch, ExpParamAnnealer
from .agentformer_lib import AgentFormerEncoderLayer, AgentFormerDecoderLayer, AgentFormerDecoder, AgentFormerEncoder
from tbsim.models.agentformer_lib import *
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.loss_utils import MultiModal_trajectory_loss
from tbsim.dynamics import DoubleIntegrator
from tbsim.models import base_models
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation, DynOrnsteinUhlenbeckPerturbation
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.loss_utils import (
    trajectory_loss,
    MultiModal_trajectory_loss,
    goal_reaching_loss,
    collision_loss,
    collision_loss_masked,
    log_normal_mixture,
    NLL_GMM_loss,
    compute_pred_loss,
    diversity_score,
)


""" Positional Encoding """
class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


""" Context (Past) Encoder """
class ContextEncoder(nn.Module):
    def __init__(self, algo_config, **kwargs):
        super().__init__()
        self.algo_config = algo_config
        self.motion_dim = algo_config['motion_dim']
        self.model_dim = algo_config['tf_model_dim']
        self.ff_dim = algo_config['tf_ff_dim']
        self.nhead = algo_config['tf_nhead']
        self.dropout = algo_config['tf_dropout']
        self.nlayer = algo_config["context_encoder"]['nlayer']
        self.input_type = algo_config['input_type']
        self.pooling = algo_config.pooling
        self.agent_enc_shuffle = algo_config['agent_enc_shuffle']
        self.vel_heading = algo_config['vel_heading']
        in_dim = self.motion_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += algo_config.map_encoder.feature_dim - self.motion_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)
        

        encoder_layers = AgentFormerEncoderLayer({}, self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=algo_config['pos_concat'], max_a_len=algo_config['max_agent_len'], use_agent_enc=algo_config['use_agent_enc'], agent_enc_learn=algo_config['agent_enc_learn'])

    def forward(self, data):
        pre_len, agent_num, bs = (
            data["pre_motion"].size(0),
            data["pre_motion"].size(1),
            data["pre_motion"].size(2),
        )
        PN = pre_len * agent_num

        # get raw features
        traj_in = []
        for key in self.input_type:
            if key == "pos":
                traj_in.append(data["pre_motion"])  # P x N x B x 2
            elif key == "vel":
                vel = data["pre_vel"]  # P x N x B x 2
                # if len(self.input_type) > 1:
                # vel = torch.cat([vel[[0]], vel], dim=0)
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data["heading"])[0]
                traj_in.append(vel)
            elif key == "norm":
                traj_in.append(data["pre_motion_norm"])  # P x N x B x 2
            elif key == "scene_norm":
                traj_in.append(data["pre_motion_scene_norm"])  # P x N x B x 2
            elif key == "heading":
                hv = (
                    data["heading_vec"].unsqueeze(0).repeat_interleave(pre_len, dim=0)
                )  # P x N x B x 2
                traj_in.append(hv)
            elif key == "map":
                map_enc = data["map_enc"].unsqueeze(0).repeat((pre_len, 1, 1, 1))
                traj_in.append(map_enc)
            else:
                raise ValueError("unknown input_type!")

        # extend the agent-pair mask to PN x PN by repeating
        # TODO: (Xinshuo) combine this mask with the time-stamp based masking
        # src_agent_mask = data['agent_mask'].clone()     # N x N
        # src_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], src_agent_mask).to(tf_in.device)    # PN X PN

        # ******************************** create mask for NaN

        # time-stamp based masking, i.e., not masking for a whole agents
        # can only mask part of the agents who have incomplete data
        src_mask = (
            data["pre_mask"].transpose(1, 2).contiguous().view(bs, PN, 1)
        )  # B x PN x 1
        src_mask_square = torch.bmm(src_mask, src_mask.transpose(1, 2))  # B x PN x PN

        # due to the inverse definition in attention.py
        # 0 means good, 1 means nan data
        enc_mask = (1 - src_mask.transpose(0, 1)).bool()  # PN x B x 1
        src_mask_square = (1 - src_mask_square).bool()  # B x PN x PN

        # expand mask to head dimensions
        src_mask_square = (
            src_mask_square.unsqueeze(1)
            .repeat_interleave(self.nhead, dim=1)
            .view(bs * self.nhead, PN, PN)
        )  # BH x PN x PN
        # repeat_interleave copy for the dimenion that already has sth, e.g., B
        # attach the copied dimenion in the end, i.e., BH rather than HB
        # the order matters in this case since there are a lot of dimenions
        # when printing the matrices, the default is to loop/list from the
        # 2nd dimenion, which is H in this case, same for PN (N dim goes first)

        # ******************************** feature encoding

        # mask NaN because even simple fc cannot handle NaN in backward pass
        traj_in = torch.cat(traj_in, dim=-1)  # P x N x B x feat
        traj_in = traj_in.view(PN, bs, traj_in.shape[-1])  # PN x B x feat
        traj_in = traj_in.masked_fill_(enc_mask, float(0))  # PN x B x feat

        # input projection
        tf_in = self.input_fc(traj_in)  # PN x B x feat
        tf_in = tf_in.masked_fill_(enc_mask, float(0.0))
        # the resulting features will contain some randome numbers in the
        # invalid rows, can suppress using the above comment
        # optional: but not masking will not affect the final results

        # ******************************** transformer

        # add positional embedding
        agent_enc_shuffle = (
            data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
        )
        tf_in_pos = self.pos_encoder(
            tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle
        )  # PN x B x feat

        tf_in_pos = tf_in_pos.masked_fill_(enc_mask, float(0.0))
        # the resulting features will contain some randome numbers in the
        # invalid rows, can suppress using the above comment
        # optional: but not masking will not affect the final results

        # transformer encoder
        assert not torch.isnan(tf_in_pos).any(), "error"
        data["context_enc"] = self.tf_encoder(
            tf_in_pos, mask=src_mask_square, num_agent=agent_num  # BH x PN x PN
        )  # PN x B x feat
        assert not torch.isnan(data["context_enc"]).any(), "error"

        # mask NaN row (now contained random numbers due to softmax and bias in the linear layers)
        # replace random numbers in the NaN rows with 0s to avoid confusion
        # here, the masking is needed, otherwise will affect the prior in the pooling
        data["context_enc"] = data["context_enc"].masked_fill_(
            enc_mask, float(0.0)
        )  # PN x B x feat

        # ******************************** compute latent distribution

        # compute per agent context for prior
        # TODO: (Xinshuo) try max pooling again with mask, to see if better performance can be obtained
        # using mean will average over a few zeros for the agents with invalid data
        # TODO: (Xinshuo) try torch.nanmean() to ignore the zeros or nan for average pooling
        context_rs = data["context_enc"].view(
            pre_len, agent_num, bs, self.model_dim
        )  # P x N x B x feat
        if self.pooling == "mean":
            data["agent_context"] = torch.mean(context_rs, dim=0)  # N x B x feat
        else:
            data["agent_context"] = torch.max(context_rs, dim=0)[0]
        data["agent_context"] = data["agent_context"].view(
            agent_num * bs, -1
        )  # NB x feat


""" Future Encoder """
class FutureEncoder(nn.Module):
    def __init__(self, algo_config, **kwargs):
        super().__init__()
        self.algo_config = algo_config
        self.context_dim = context_dim = algo_config['tf_model_dim']
        self.forecast_dim = forecast_dim = algo_config['forecast_dim']
        self.nz = algo_config['nz']
        self.z_type = algo_config['z_type']

        self.model_dim = algo_config['tf_model_dim']
        self.ff_dim = algo_config['tf_ff_dim']
        self.nhead = algo_config['tf_nhead']
        self.dropout = algo_config['tf_dropout']
        self.nlayer = algo_config["future_encoder"]["nlayer"]
        self.out_mlp_dim = algo_config.future_decoder.out_mlp_dim
        self.input_type = algo_config['fut_input_type']
        self.pooling = algo_config.pooling
        self.agent_enc_shuffle = algo_config.agent_enc_shuffle
        self.vel_heading = algo_config.vel_heading
        # networks
        in_dim = forecast_dim * len(self.input_type)
        if 'map' in self.input_type:
            in_dim += algo_config.map_encoder.feature_dim - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer({}, self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)
        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=algo_config['pos_concat'], max_a_len=algo_config['max_agent_len'], use_agent_enc=algo_config['use_agent_enc'], agent_enc_learn=algo_config['agent_enc_learn'])
        num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
        if self.out_mlp_dim is None:
            self.q_z_net = nn.Linear(self.model_dim, num_dist_params)
        else:
            self.out_mlp = AFMLP(self.model_dim, self.out_mlp_dim, 'relu')
            self.q_z_net = nn.Linear(self.out_mlp.out_dim, num_dist_params)
        # initialize
        initialize_weights(self.q_z_net.modules())

    def forward(self, data, reparam=True,temp=0.1):
        fut_len, agent_num, bs = (
            data["fut_motion"].size(0),
            data["fut_motion"].size(1),
            data["fut_motion"].size(2),
        )
        pre_len = data["pre_motion"].size(0)
        FN = fut_len * agent_num
        PN = pre_len * agent_num

        # get input feature
        traj_in = []
        for key in self.input_type:
            if key == "pos":
                traj_in.append(data["fut_motion"])  # F x N x B x 2
            elif key == "vel":
                vel = data["fut_vel"]  # F x N x B x 2
                if self.vel_heading:
                    vel = rotation_2d_torch(vel, -data["heading"])[0]
                traj_in.append(vel)
            elif key == "norm":
                traj_in.append(data["fut_motion_norm"])  # F x N x B x 2
            elif key == "scene_norm":
                traj_in.append(data["fut_motion_scene_norm"])  # F x N x B x 2
            elif key == "heading":
                hv = (
                    data["heading_vec"].unsqueeze(0).repeat_interleave(fut_len, dim=0)
                )  # F x N x B x 2
                traj_in.append(hv)
            elif key == "map":
                map_enc = (
                    data["map_enc"]
                    .unsqueeze(0)
                    .repeat((data["fut_motion"].shape[0], 1, 1))
                )
                traj_in.append(map_enc)
            else:
                raise ValueError("unknown input_type!")

        # ******************************** create mask for NaN

        # generate masks, mem_mask for cross attention between past and future, tgt_mask for self_attention between futures
        # mem_agent_mask = data['agent_mask'].clone()     # N x N
        # mem_mask = generate_mask(tf_in.shape[0], data['context_enc'].shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)  # FN x PN
        # tgt_agent_mask = data['agent_mask'].clone()     # N x N
        # tgt_mask = generate_mask(tf_in.shape[0], tf_in.shape[0], data['agent_num'], tgt_agent_mask).to(tf_in.device)                # FN x FN

        # time-stamp based masking, i.e., not masking for a whole agents
        # can only mask part of the agents who have incomplete data
        fut_mask = (
            data["fut_mask"].transpose(1, 2).contiguous().view(bs, FN, 1)
        )  # B x FN x 1
        pre_mask = (
            data["pre_mask"].transpose(1, 2).contiguous().view(bs, PN, 1)
        )  # B x PN x 1
        mem_mask = torch.bmm(fut_mask, pre_mask.transpose(1, 2))  # B x FN x PN
        tgt_mask = torch.bmm(fut_mask, fut_mask.transpose(1, 2))  # B x FN x FN

        # due to the inverse definition in attention.py
        # 0 means good, 1 means nan data
        enc_mask = (1 - fut_mask.transpose(0, 1)).bool()  # FN x B x 1
        mem_mask = (1 - mem_mask).bool()  # B x FN x PN
        tgt_mask = (1 - tgt_mask).bool()  # B x FN x FN

        # expand mask to head dimensions
        mem_mask = (
            mem_mask.unsqueeze(1)
            .repeat_interleave(self.nhead, dim=1)
            .view(bs * self.nhead, FN, PN)
        )  # BH x FN x PN
        tgt_mask = (
            tgt_mask.unsqueeze(1)
            .repeat_interleave(self.nhead, dim=1)
            .view(bs * self.nhead, FN, FN)
        )  # BH x FN x FN

        # ******************************** feature encoding

        # mask NaN because even simple fc cannot handle NaN in backward pass
        traj_in = torch.cat(traj_in, dim=-1)  # F x N x B x feat
        traj_in = traj_in.view(FN, bs, traj_in.shape[-1])  # FN x B x feat
        traj_in = traj_in.masked_fill_(enc_mask, float(0))  # FN x B x feat

        # input projection
        tf_in = self.input_fc(traj_in)  # FN x B x feat
        tf_in = tf_in.masked_fill_(enc_mask, float(0.0))  # FN x B x feat
        # the resulting features will contain some randome numbers in the
        # invalid rows, can suppress using the above comment
        # optional: but not masking will not affect the final results

        # ******************************** transformer

        # add positional embedding
        agent_enc_shuffle = (
            data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
        )
        tf_in_pos = self.pos_encoder(
            tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle
        )  # FN x B x feat
        tf_in_pos = tf_in_pos.masked_fill_(enc_mask, float(0.0))
        # the resulting features will contain some randome numbers in the
        # invalid rows, can suppress using the above comment
        # optional: but not masking will not affect the final results

        # transformer decoder (cross attention between future and context features)
        assert not torch.isnan(tf_in_pos).any(), "error"
        tf_out, _ = self.tf_decoder(
            tf_in_pos,  # FN x B x feat
            data["context_enc"],  # PN x B x feat
            memory_mask=mem_mask,  # BH x FN x PN
            tgt_mask=tgt_mask,  # BH x FN x FN
            num_agent=agent_num,
        )  # FN x B x feat
        assert not torch.isnan(tf_out).any(), "error"

        # mask NaN row (now contained random numbers due to softmax and bias in the linear layers)
        # replace random numbers in the NaN rows with 0s to avoid confusion
        # here, the masking is needed, otherwise will affect the posterior in the pooling
        tf_out = tf_out.masked_fill_(enc_mask, float(0.0))  # FN x B x feat

        # ******************************** compute latent distribution

        # compute per agent for posterior
        tf_out = tf_out.view(fut_len, agent_num, bs, self.model_dim)  # F x N x B x feat
        if self.pooling == "mean":
            h = torch.mean(tf_out, dim=0)  # N x B x feat
        else:
            h = torch.max(tf_out, dim=0)[0]  # N x B x feat
        if self.out_mlp_dim is not None:
            h = self.out_mlp(h)  # N x B x feat
        h = h.view(agent_num * bs, -1)  # NB x feat

        # ******************************** sample latent code

        # sample latent code from the posterior distribution
        # each agent has a separate distribution and sample independently
        q_z_params = self.q_z_net(h)  # NB x 64 (contain mu and var)
        if self.z_type == "gaussian":
            data["q_z_dist"] = Normal(params=q_z_params)
        else:
            data["q_z_dist"] = Categorical(
                logits=q_z_params, temp=temp
            )
        data["q_z_samp"] = (
            data["q_z_dist"].rsample().reshape(agent_num, bs, -1)
        )  # N x B x 32


""" Future Decoder """
class FutureDecoder(nn.Module):
    def __init__(self, algo_config, **kwargs):
        super().__init__()
        self.algo_config = algo_config
        self.ar_detach = algo_config['ar_detach']
        self.context_dim = context_dim = algo_config['tf_model_dim']
        self.forecast_dim = forecast_dim = algo_config['forecast_dim']
        self.pred_scale = algo_config["pred_scale"]
        self.pred_type = algo_config['pred_type']
        self.sn_out_type = algo_config['sn_out_type']
        self.sn_out_heading = algo_config['sn_out_heading']
        self.input_type = algo_config['dec_input_type']
        self.future_frames = algo_config['future_num_frames']
        self.past_frames = algo_config['history_num_frames']
        self.nz = algo_config['nz']
        self.z_type = algo_config['z_type']
        self.model_dim = algo_config['tf_model_dim']
        self.ff_dim = algo_config['tf_ff_dim']
        self.nhead = algo_config['tf_nhead']
        self.dropout = algo_config['tf_dropout']
        self.nlayer = algo_config["future_decoder"]["nlayer"]
        self.out_mlp_dim = algo_config.future_decoder.out_mlp_dim
        self.pos_offset = algo_config.pos_offset
        self.agent_enc_shuffle = algo_config['agent_enc_shuffle']
        self.learn_prior = algo_config['learn_prior']
        # networks
        if self.pred_type in ["dynamic","dynamic_var"]:
            in_dim = 6+len(self.input_type) * forecast_dim + self.nz

            if algo_config.dynamic_type=="Unicycle":
                self.dyn = dynamics.Unicycle()
            else:
                raise Exception("not supported dynamic type")
        else:
            in_dim = forecast_dim + len(self.input_type) * forecast_dim + self.nz
        if 'map' in self.input_type:
            in_dim += algo_config.map_encoder.feature_dim - forecast_dim
        self.input_fc = nn.Linear(in_dim, self.model_dim)

        decoder_layers = AgentFormerDecoderLayer({}, self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.tf_decoder = AgentFormerDecoder(decoder_layers, self.nlayer)

        self.pos_encoder = PositionalAgentEncoding(self.model_dim, self.dropout, concat=algo_config['pos_concat'], max_a_len=algo_config['max_agent_len'], use_agent_enc=algo_config['use_agent_enc'], agent_enc_learn=algo_config['agent_enc_learn'])
        if self.pred_type in ["scene_norm","vel","pos","dynamic"]:
            outdim = forecast_dim
        elif self.pred_type == "dynamic_var":
            outdim = forecast_dim+2
        if self.out_mlp_dim is None:
            self.out_fc = nn.Linear(self.model_dim, outdim)
        else:
            in_dim = self.model_dim
            self.out_mlp = AFMLP(in_dim, self.out_mlp_dim, 'relu')
            self.out_fc = nn.Linear(self.out_mlp.out_dim, outdim)
        initialize_weights(self.out_fc.modules())
        if self.learn_prior:
            num_dist_params = 2 * self.nz if self.z_type == 'gaussian' else self.nz     # either gaussian or discrete
            self.p_z_net = nn.Linear(self.model_dim, num_dist_params)
            initialize_weights(self.p_z_net.modules())
    def decode_traj_ar_orig(
        self,
        data,
        mode,
        context,
        pre_motion,
        pre_vel,
        pre_motion_scene_norm,
        z,
        sample_num,
        need_weights=False,
    ):
        # z: N x BS x 32

        fut_len, agent_num, bs = (
            data["fut_motion"].size(0),
            data["fut_motion"].size(1),
            data["fut_motion"].size(2),
        )
        pre_len = data["pre_motion"].size(0)
        FN = fut_len * agent_num
        PN = pre_len * agent_num

        # get input feature, only take the current timestamp as input here
        if self.pred_type == "vel":
            dec_in = pre_vel[[-1]]  # 1 x N x BS x 2
        elif self.pred_type == "pos":
            dec_in = pre_motion[[-1]]  # 1 x N X BS x 2
        elif self.pred_type == "scene_norm":
            dec_in = pre_motion_scene_norm[[-1]]  # 1 x N x BS x 2
        else:
            dec_in = torch.zeros_like(pre_motion[[-1]])  # 1 x N x BS x 2

        # concatenate conditional input features with latent code
        # broadcast to the sample dimension
        dec_in = dec_in.view(
            agent_num, bs * sample_num, dec_in.size(-1)
        )  # N x BS x feat
        in_arr = [dec_in, z]

        # add additional features such as the map
        for key in self.input_type:
            if key == "heading":
                heading = data["heading_vec"].repeat_interleave(
                    sample_num, dim=1
                )  # N x BS x 2

                in_arr.append(heading)
            elif key == "map":
                map_enc = data["map_enc"].unsqueeze(1).repeat((1, sample_num, 1))
                in_arr.append(map_enc)
            else:
                raise ValueError("wrong decode input type!")
        dec_in_z = torch.cat(in_arr, dim=-1)  # (1)N x BS x feat

        # TODO: add this agent-wise mask back later for introducing non-fully-connected attention
        # mem_agent_mask = data['agent_mask'].clone()
        # tgt_agent_mask = data['agent_mask'].clone()

        # predict for each timestamps auto-regressively
        for fut_index in range(fut_len):
            F_tmp = fut_index + 1
            FN_tmp = F_tmp * agent_num

            # ******************************** create mask for NaN

            # agent-wise masking
            # mem_mask = pred_utils.generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)   # (F)N x PN
            # tgt_mask = pred_utils.generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)      # (F)N x (F)N

            # time-stamp-based masking
            # only using the last timestamp of pre_motion, i.e., the current frame of mask
            # repeat it over future frames, i.e., the assumption is that the valid objects
            # to predict must have data in the current frame, this is safe since we interpolated
            # data in the trajdata, i.e., objects with incomplete trajectories may have NaN in the
            # beginning/end of the time window, but not in the current frame
            cur_mask = (
                data["pre_mask"][:, :, [-1]]
                .transpose(1, 2)
                .contiguous()
                .view(bs, agent_num, 1)
            )  # B x N x 1
            cur_mask = cur_mask.repeat_interleave(sample_num, dim=0)  # BS x N x 1
            cur_mask = (
                cur_mask.unsqueeze(1)
                .repeat_interleave(F_tmp, dim=1)
                .view(bs * sample_num, FN_tmp, 1)
            )  # BS x (F)N x 1
            
            pre_mask = (
                data["pre_mask"]
                .transpose(1, 2)
                .contiguous()
                .view(bs, PN, 1)
                .repeat_interleave(sample_num, dim=0)
            )  # BS x PN x 1
            mem_mask = torch.bmm(cur_mask, pre_mask.transpose(1, 2))  # BS x (F)N x PN
            tgt_mask = torch.bmm(cur_mask, cur_mask.transpose(1, 2))  # BS x (F)N x (F)N
            

            # due to the inverse definition in attention.py
            # 0 means good, 1 means nan data now
            cur_mask = (1 - cur_mask.transpose(0, 1)).bool()  # (F)N x BS x 1
            mem_mask = (1 - mem_mask).bool()  # BS x (F)N x PN
            tgt_mask = (1 - tgt_mask).bool()  # BS x (F)N x (F)N

            # expand mask to head dimensions
            mem_mask = (
                mem_mask.unsqueeze(1)
                .repeat_interleave(self.nhead, dim=1)
                .view(bs*sample_num*self.nhead, FN_tmp, PN)
            )  # BSH x (F)N x PN
            tgt_mask = (
                tgt_mask.unsqueeze(1)
                .repeat_interleave(self.nhead, dim=1)
                .view(bs*sample_num*self.nhead, FN_tmp, FN_tmp)
            )  # BSH x (F)N x (F)N

            # ******************************** feature encoding

            # mask NaN because even simple fc cannot handle NaN in backward pass
            tf_in = dec_in_z.masked_fill_(cur_mask, float(0))  # (F)N x BS x feat

            # input projection
            tf_in = self.input_fc(
                tf_in
            )  # (F)N x BS x feat, F is increamentally increased
            
            # optional: not masking will not affect the final results
            # just to suppress some random numbers generated by linear layer's bias
            # for cleaner printing, but these random numbers are not used later due to masking
            tf_in = tf_in.masked_fill_(cur_mask, float(0.0))  # (F)N x BS x feat

            # ******************************** transformer

            # add positional encoding
            agent_enc_shuffle = (
                data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
            )
            tf_in_pos = self.pos_encoder(
                tf_in,
                num_a=agent_num,
                agent_enc_shuffle=agent_enc_shuffle,
                t_offset=self.past_frames - 1 if self.pos_offset else 0,
            )
            # (F)N x BS x feat, F is increamentally increased

            # optional: not masking will not affect the final results
            # just to suppress some random numbers generated by linear layer's bias
            # for cleaner printing, but these random numbers are not used later due to masking
            tf_in_pos = tf_in_pos.masked_fill_(cur_mask, float(0.0))

            # transformer decoder (between predicted steps and past context)
            assert not torch.isnan(tf_in_pos).any(), "error"
            tf_out, attn_weights = self.tf_decoder(
                tf_in_pos,  # (F)N x BS x feat
                context,  # PN x BS x feat
                memory_mask=mem_mask,  # BSH x (F)N x PN
                tgt_mask=tgt_mask,  # BSH x (F)N x (F)N
                num_agent=agent_num,
                need_weights=need_weights,
            )
            assert not torch.isnan(tf_out).any(), "error"
            # tf_out: (F)N x BS x feat

            # ******************************** output projection

            # convert the output feature to output dimension (x, y)
            # out_tmp = tf_out.view(-1, tf_out.shape[-1])  # (F)NS x feat
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(tf_out)  # (F)N x BS x feat
            seq_out = self.out_fc(out_tmp)  # (F)N x BS x 2

            # denormalize data and de-rotate
            if self.pred_type == "scene_norm" and self.sn_out_type in {"vel", "norm"}:
                norm_motion = seq_out.view(
                    F_tmp, agent_num, bs * sample_num, seq_out.shape[-1]
                )  # (F) x N x BS x 2

                # aggregate velocity prediction to obtain location
                if self.sn_out_type == "vel":
                    norm_motion = torch.cumsum(norm_motion, dim=0)  # (F) x N x BS x 2

                # default not used
                if self.sn_out_heading:
                    angles = data["heading"].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]

                # denormalize over the scene
                # we are predicting delta with respect to the current frame of data
                # will introduce NaN here since the scene_norm data in the current frame has NaN
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]  # (F) x N x BS x 2
                seq_out = seq_out.view(
                    FN_tmp, bs * sample_num, seq_out.shape[-1]
                )  # (F)N x BS x 2

            # ******************************** prepare for the next timestamp

            # only take the last few results for the N agents predicted in the last timestamp
            # TODO: wasted resourses on re-computing some timestamps
            if self.ar_detach:
                out_in = seq_out[-agent_num:].clone().detach()  # N x BS x 2
            else:
                out_in = seq_out[-agent_num:]  # N x BS x 2

            # create input for the next timestamp
            in_arr = [out_in, z]  # z: N x BS x 32
            for key in self.input_type:
                if key == "heading":
                    in_arr.append(heading)  # z: N x BS x 2
                elif key == "map":
                    in_arr.append(map_enc)
                else:
                    raise ValueError("wrong decoder input type!")

            # combine with previous information, data in normal forward order
            # i.e., newly predicted information attached in the end of features
            # TODO: (Xinshuo) z is combined twice here, redundancy
            out_in_z = torch.cat(in_arr, dim=-1)                # N x BS x feat
            dec_in_z = torch.cat([dec_in_z, out_in_z], dim=0)   # (F+1)N x BS x feat

        # seq_out: FN x BS x 2
        seq_out = seq_out.view(
            fut_len, agent_num,  bs*sample_num, seq_out.shape[-1]
        )  # F x N x BS x 2
        data[f"{mode}_seq_out"] = seq_out

        if self.pred_type == "vel":
            dec_motion = torch.cumsum(seq_out, dim=0)   # F x N x BS x 2
            dec_motion += pre_motion[[-1]]              # F x N X BS x 2
        elif self.pred_type == "pos":
            dec_motion = seq_out.clone()
        elif self.pred_type == "scene_norm":
            dec_motion = seq_out + data["scene_orig"].repeat_interleave(sample_num, dim=0)  # F x N X BS x 2
        else:
            dec_motion = seq_out + pre_motion[[-1]]     # F x N X BS x 2

        # reshape for loss computation
        dec_motion = dec_motion.transpose(0, 2).contiguous()  # BS x N x F x 2
        if mode == "infer":
            dec_motion = dec_motion.view(
                bs, sample_num, agent_num, fut_len, dec_motion.size(-1)
            )  # B x S x N x F x 2

        data[f"{mode}_dec_motion"] = dec_motion
        if need_weights:
            data["attn_weights"] = attn_weights
    def decode_traj_ar(
        self,
        data,
        mode,
        context,
        input_dict,
        z,
        sample_num,
        need_weights=False,
        cond_idx=None,
    ):
        # z: N x BS x 32

        fut_len, agent_num, bs = (
            data["fut_motion"].size(0),
            data["fut_motion"].size(1),
            data["fut_motion"].size(2),
        )
        pre_len = data["pre_motion"].size(0)
        FN = fut_len * agent_num
        PN = pre_len * agent_num
        device = data["fut_motion"]
        # get input feature, only take the current timestamp as input here
        if self.pred_type == "vel":
            pre_vel = input_dict["pre_vel"]
            fut_vel = input_dict["fut_vel"]
            dec_in = torch.cat((pre_vel[[-1]],fut_vel))  # (1+F) x N x BS x 2
        elif self.pred_type == "pos":
            pre_motion = input_dict["pre_motion"]
            fut_motion = input_dict["fut_motion"]
            dec_in = torch.cat((pre_motion[[-1]],fut_motion),0)  # (1+F) x N X BS x 2
        elif self.pred_type == "scene_norm":
            pre_motion_scene_norm = input_dict["pre_motion_scene_norm"]
            fut_motion_scene_norm = input_dict["fut_motion_scene_norm"]
            dec_in = torch.cat((pre_motion_scene_norm[[-1]],fut_motion_scene_norm),0)  # (1+F) x N x BS x 2
        elif self.pred_type =="dynamic":
            curr_state = input_dict["curr_state"]
            pre_state_vec = input_dict["pre_state_vec"]
            fut_state_vec = input_dict["fut_state_vec"]
            dec_in = torch.cat((pre_state_vec[[-1]],fut_state_vec)) # (1+F) x N x BS x 6
            dec_state = [curr_state]
        elif self.pred_type == "dynamic_var":
            curr_state = input_dict["curr_state"]
            pre_state_vec = input_dict["pre_state_vec"]
            fut_state_vec = input_dict["fut_state_vec"]
            dec_in = torch.cat((pre_state_vec[[-1]],fut_state_vec)) # (1+F) x N x BS x 6
            dec_state = [curr_state]

        else:
            dec_in = torch.zeros([1+fut_len,agent_num,bs*sample_num,2]).to(device)  # (1+F) x N x BS x 2

        # concatenate conditional input features with latent code
        # broadcast to the sample dimension
        
        z_tiled = z.unsqueeze(0).repeat_interleave(1+fut_len,0)

        dec_in = dec_in.view(
            (1+fut_len)*agent_num, bs * sample_num, dec_in.size(-1)
        )  # (1+F)N x BS x feat
        in_arr = [dec_in, TensorUtils.join_dimensions(z_tiled,0,2)]

        # add additional features such as the map
        for key in self.input_type:
            if key == "heading":
                heading = data["heading_vec"].repeat_interleave(
                    sample_num, dim=1
                )  # N x BS x 2
                heading_tiled = heading.repeat(1+fut_len,1,1)

                in_arr.append(heading_tiled)
            elif key == "map":
                map_enc = data["map_enc"].repeat_interleave(sample_num,1)
                map_enc_tiled = map_enc.repeat(1+fut_len,1,1)
                in_arr.append(map_enc_tiled)
            else:
                raise ValueError("wrong decode input type!")
        dec_in_z_orig = torch.cat(in_arr, dim=-1)  # (1)N x BS x feat
        D = dec_in_z_orig.shape[-1]
        device = dec_in.device
        orig_dec_in_z_list = list(torch.split(dec_in_z_orig,agent_num))
        updated_dec_in_z_list = list()
        dec_in_z = dec_in_z_orig.clone()

        # dec_in_z_padded = torch.cat((dec_in_z,torch.zeros(agent_num*(fut_len-1),bs,D).to(device)))
        

        # TODO: add this agent-wise mask back later for introducing non-fully-connected attention
        # mem_agent_mask = data['agent_mask'].clone()
        # tgt_agent_mask = data['agent_mask'].clone()
        
        if self.pred_type == "dynamic_var":
            logvar = list()
        
        # predict for each timestamps auto-regressively
        for fut_index in range(fut_len):
            F_tmp = fut_index + 1
            FN_tmp = F_tmp * agent_num

            # ******************************** create mask for NaN

            # agent-wise masking
            # mem_mask = pred_utils.generate_mask(tf_in.shape[0], context.shape[0], data['agent_num'], mem_agent_mask).to(tf_in.device)   # (F)N x PN
            # tgt_mask = pred_utils.generate_ar_mask(tf_in_pos.shape[0], agent_num, tgt_agent_mask).to(tf_in.device)      # (F)N x (F)N

            # time-stamp-based masking
            # only using the last timestamp of pre_motion, i.e., the current frame of mask
            # repeat it over future frames, i.e., the assumption is that the valid objects
            # to predict must have data in the current frame, this is safe since we interpolated
            # data in the trajdata, i.e., objects with incomplete trajectories may have NaN in the
            # beginning/end of the time window, but not in the current frame
            cur_mask = (
                data["pre_mask"][:, :, [-1]]
                .transpose(1, 2)
                .contiguous()
                .view(bs, agent_num, 1)
            )  # B x N x 1
            cur_mask = cur_mask.repeat_interleave(sample_num, dim=0)  # BS x N x 1
            cur_mask = cur_mask.unsqueeze(1).repeat_interleave(1+fut_len, dim=1)

            cur_mask[:,F_tmp:] = 0
            if cond_idx is not None:
                cur_mask[:,:,cond_idx]=1

            cur_mask = cur_mask.view(bs * sample_num, (1+fut_len)*agent_num, 1)

            pre_mask = (
                data["pre_mask"]
                .transpose(1, 2)
                .contiguous()
                .view(bs, PN, 1)
                .repeat_interleave(sample_num, dim=0)
            )  # BS x PN x 1
            
            mem_mask = torch.bmm(cur_mask, pre_mask.transpose(1, 2))  # BS x (1+F)N x PN
            tgt_mask = torch.bmm(cur_mask, cur_mask.transpose(1, 2))  # BS x (1+F)N x (1+F)N
            

            # due to the inverse definition in attention.py
            # 0 means good, 1 means nan data now
            cur_mask = (1 - cur_mask.transpose(0, 1)).bool()  # (1+F)N x BS x 1
            mem_mask = (1 - mem_mask).bool()  # BS x (1+F)N x PN
            tgt_mask = (1 - tgt_mask).bool()  # BS x (1+F)N x (1+F)N

            # expand mask to head dimensions
            mem_mask = (
                mem_mask.unsqueeze(1)
                .repeat_interleave(self.nhead, dim=1)
                .view(bs*sample_num*self.nhead, (1+fut_len)*agent_num, PN)
            )  # BSH x (1+F)N x PN
            tgt_mask = (
                tgt_mask.unsqueeze(1)
                .repeat_interleave(self.nhead, dim=1)
                .view(bs*sample_num*self.nhead, (1+fut_len)*agent_num, (1+fut_len)*agent_num)
            )  # BSH x (1+F)N x (1+F)N

            # ******************************** feature encoding

            # mask NaN because even simple fc cannot handle NaN in backward pass
            
            tf_in = dec_in_z.masked_fill_(cur_mask, float(0))  # (1+F)N x BS x feat

            # input projection
            tf_in = self.input_fc(
                tf_in
            )  # (F)N x BS x feat, F is increamentally increased
            
            # optional: not masking will not affect the final results
            # just to suppress some random numbers generated by linear layer's bias
            # for cleaner printing, but these random numbers are not used later due to masking
            tf_in = tf_in.masked_fill_(cur_mask, float(0.0))  # (1+F)N x BS x feat

            # ******************************** transformer

            # add positional encoding
            agent_enc_shuffle = (
                data["agent_enc_shuffle"] if self.agent_enc_shuffle else None
            )
            tf_in_pos = self.pos_encoder(
                tf_in,
                num_a=agent_num,
                agent_enc_shuffle=agent_enc_shuffle,
                t_offset=self.past_frames - 1 if self.pos_offset else 0,
            )
            # (F)N x BS x feat, F is increamentally increased

            # optional: not masking will not affect the final results
            # just to suppress some random numbers generated by linear layer's bias
            # for cleaner printing, but these random numbers are not used later due to masking
            tf_in_pos = tf_in_pos.masked_fill_(cur_mask, float(0.0))

            # transformer decoder (between predicted steps and past context)
            assert not torch.isnan(tf_in_pos).any(), "error"
            
            tf_out, attn_weights = self.tf_decoder(
                tf_in_pos,  # (F)N x BS x feat
                context,  # PN x BS x feat
                memory_mask=mem_mask,  # BSH x (F)N x PN
                tgt_mask=tgt_mask,  # BSH x (F)N x (F)N
                num_agent=agent_num,
                need_weights=need_weights,
            )

            assert not torch.isnan(tf_out).any(), "error"
            # tf_out: (1+F)N x BS x feat


            # ******************************** output projection

            # convert the output feature to output dimension (x, y)
            # out_tmp = tf_out.view(-1, tf_out.shape[-1])  # (F)NS x feat
            if self.out_mlp_dim is not None:
                out_tmp = self.out_mlp(tf_out)  # (F)N x BS x feat
            seq_out = self.out_fc(out_tmp)  # (F)N x BS x 2

            # denormalize data and de-rotate
            if self.pred_type == "scene_norm" and self.sn_out_type in {"vel", "norm"}:
                norm_motion = seq_out.view(
                    1+fut_len, agent_num, bs * sample_num, seq_out.shape[-1]
                )  # (1+F) x N x BS x 2

                # aggregate velocity prediction to obtain location
                if self.sn_out_type == "vel":
                    norm_motion = torch.cumsum(norm_motion, dim=0)  # (1+F) x N x BS x 2

                # default not used
                if self.sn_out_heading:
                    angles = data["heading"].repeat_interleave(sample_num)
                    norm_motion = rotation_2d_torch(norm_motion, angles)[0]

                # denormalize over the scene
                # we are predicting delta with respect to the current frame of data
                # will introduce NaN here since the scene_norm data in the current frame has NaN
                seq_out = norm_motion + pre_motion_scene_norm[[-1]]  # (1+F) x N x BS x 2
                dec_feat_in = seq_out.view(
                    (1+fut_len)*agent_num, bs * sample_num, seq_out.shape[-1]
                )  # (1+F)N x BS x 2
            elif self.pred_type in ["dynamic","dynamic_var"]:
                traj_scale = data["traj_scale"]
                input_seq = TensorUtils.reshape_dimensions_single(seq_out[...,:self.forecast_dim],0,1,[fut_len+1,-1]).permute(1,2,0,3)
                state_seq,_,_ = self.dyn.forward_dynamics(curr_state,input_seq[...,1:,:],data["step_time"])
                state_seq = state_seq.permute(2,0,1,3)
                state_seq = torch.cat((curr_state.unsqueeze(0),state_seq),0)
                yaw = state_seq[...,3:]
                vel = state_seq[...,2:3]/traj_scale
                cosyaw = torch.cos(yaw)
                sinyaw = torch.sin(yaw)
                dec_feat_in = TensorUtils.join_dimensions(torch.cat((state_seq[...,:2]/traj_scale,vel*cosyaw,vel*sinyaw,cosyaw,sinyaw),-1),0,2)
                

            # ******************************** prepare for the next timestamp

            # only take the last few results for the N agents predicted in the last timestamp
            # TODO: wasted resourses on re-computing some timestamps
            if self.ar_detach:
                out_in = dec_feat_in[F_tmp*agent_num:(1+F_tmp)*agent_num].clone().detach()  # N x BS x 2(6)
            else:
                out_in = dec_feat_in[F_tmp*agent_num:(1+F_tmp)*agent_num]  # N x BS x 2(6)

            # create input for the next timestamp
            in_arr = [out_in, z]  # z: N x BS x 32

            for key in self.input_type:
                if key == "heading":
                    in_arr.append(heading)  # z: N x BS x 2
                elif key == "map":
                    in_arr.append(map_enc)
                else:
                    raise ValueError("wrong decoder input type!")

            # combine with previous information, data in normal forward order
            # i.e., newly predicted information attached in the end of features
            # TODO: (Xinshuo) z is combined twice here, redundancy
            out_in_z = torch.cat(in_arr, dim=-1)                # N x BS x feat
            updated_dec_in_z_list.append(out_in_z)
            # import pdb
            # pdb.set_trace()
            curr_dec_list = orig_dec_in_z_list[0:1]+updated_dec_in_z_list+orig_dec_in_z_list[F_tmp+1:]
            dec_in_z = torch.cat(curr_dec_list,0)
            # dec_in_z[F_tmp*agent_num:(1+F_tmp)*agent_num] = out_in_z

        
        # seq_out: FN x BS x 2
        seq_out = seq_out.view(
            1+fut_len, agent_num,  bs*sample_num, seq_out.shape[-1]
        )  # 1+F x N x BS x 2
        seq_out = seq_out[1:] #F x N x BS x 2
        data[f"{mode}_seq_out"] = seq_out

        if self.pred_type == "vel":
            dec_motion = torch.cumsum(seq_out, dim=0)   # F x N x BS x 2
            dec_motion += pre_motion[[-1]]              # F x N X BS x 2
        elif self.pred_type == "pos":
            dec_motion = seq_out.clone()
        elif self.pred_type == "scene_norm":
            dec_motion = seq_out + data["scene_orig"].repeat_interleave(sample_num, dim=0)  # F x N X BS x 2
        elif self.pred_type in ["dynamic","dynamic_var"]:
            
            input_seq = seq_out.permute(1,2,0,3)
            state_seq,_,_ = self.dyn.forward_dynamics(curr_state,input_seq[...,:self.forecast_dim],data["step_time"])
            state_seq = state_seq.permute(2,0,1,3)
            dec_state = state_seq
            dec_motion = state_seq[...,:2]/data["traj_scale"]                

        else:
            dec_motion = seq_out + pre_motion[[-1]]     # F x N X BS x 2

        # reshape for loss computation
        dec_motion = dec_motion.transpose(0, 2).contiguous()  # BS x N x F x 2
        if mode == "infer":
            dec_motion = dec_motion.view(
                bs, sample_num, agent_num, fut_len, dec_motion.size(-1)
            )  # B x S x N x F x 2
            if self.pred_type in["dynamic","dynamic_var"]:
                dec_state = dec_state.transpose(0, 2).contiguous().view(
                    bs, sample_num, agent_num, fut_len, dec_state.size(-1)
                )
                data[f"{mode}_dec_state"] = dec_state
            if self.pred_type=="dynamic_var":
                logvar = seq_out[...,self.forecast_dim:2*self.forecast_dim]
                var = torch.exp(logvar)*data["traj_scale"]**2
                var = var.permute(2,1,0,3).contiguous().view(
                    bs, sample_num, agent_num, fut_len, var.size(-1)
                )
                data[f"{mode}_var"] = var

        data[f"{mode}_dec_motion"] = dec_motion
        if need_weights:
            data["attn_weights"] = attn_weights

    def decode_traj_batch(
        self,
        data,
        mode,
        context,
        input_dict,
        z,
        sample_num,
    ):
        raise NotImplementedError

    def forward(
        self, data, mode, sample_num=1, autoregress=True, z=None, need_weights=False, cond_idx=None,temp=0.1, predict=False
    ):
        agent_num, bs = (
            data["fut_motion"].size(1),
            data["fut_motion"].size(2),
        )

        # conditional input to the decoding process
        context = data["context_enc"].repeat_interleave(
            sample_num, dim=1
        )  # PN x BS x feat

        pre_motion = data["pre_motion"].repeat_interleave(
            sample_num, dim=2
        )  # P x N X BS x 2
        fut_motion = data["fut_motion"].repeat_interleave(
            sample_num, dim=2
        )  # F x N X BS x 2
        pre_motion_scene_norm = data["pre_motion_scene_norm"].repeat_interleave(
            sample_num, dim=2
        )  # P x N x BS x 2
        fut_motion_scene_norm = data["fut_motion_scene_norm"].repeat_interleave(
            sample_num, dim=2
        )  # F x N x BS x 2
        input_dict = dict(pre_motion=pre_motion,fut_motion=fut_motion,\
                          pre_motion_scene_norm=pre_motion_scene_norm,fut_motion_scene_norm=fut_motion_scene_norm)
        if self.pred_type == "vel":
            input_dict["pre_vel"] = data["pre_vel"].repeat_interleave(sample_num, dim=2) # P x N x BS x 2
            input_dict["fut_vel"] = data["fut_vel"].repeat_interleave(sample_num, dim=2) # F x N x BS x 2
        elif self.pred_type in ["dynamic","dynamic_var"]:
            traj_scale=data["traj_scale"]
            pre_state = torch.cat((data["pre_motion"]*traj_scale,torch.norm(data["pre_vel"],dim=-1,keepdim=True)*traj_scale,
                                    data["pre_heading_raw"].transpose(0,2).unsqueeze(-1)),-1)  # P x N x B x 4 (unscaled)
            
            pre_state_vec = torch.cat((data["pre_motion"],data["pre_vel"],data["pre_heading_vec"]),-1) # P x N x B x 6 (scaled)
            fut_state_vec = torch.cat((data["fut_motion"],data["fut_vel"],data["fut_heading_vec"]),-1) # F x N x B x 6 (scaled)
            input_dict["curr_state"] = pre_state[-1].repeat_interleave(sample_num,dim=1)
            input_dict["pre_state_vec"] = pre_state_vec.repeat_interleave(sample_num, dim=2)
            input_dict["fut_state_vec"] = fut_state_vec.repeat_interleave(sample_num, dim=2)

            

        # p(z), compute prior distribution
        if mode=="infer":
            prior_key = "p_z_dist_infer"
        else:
            prior_key = "q_z_dist" if "q_z_dist" in data else "p_z_dist"
       
        if self.learn_prior:
            p_z_params0 = self.p_z_net(data["agent_context"])
            
            h = data["agent_context"].repeat_interleave(sample_num, dim=0)  # NBS x feat
            p_z_params = self.p_z_net(h)  # NBS x 64
            if self.z_type == "gaussian":
                data["p_z_dist_infer"] = Normal(params=p_z_params)
                data["p_z_dist"] = Normal(params=p_z_params0)
            else:
                data["p_z_dist_infer"] = Categorical(logits=p_z_params,temp=temp)
                data["p_z_dist"] = Categorical(logits=p_z_params0,temp=temp)
        else:
            if self.z_type == "gaussian":
                data[prior_key] = Normal(
                    mu=torch.zeros(pre_motion.shape[1], self.nz).to(pre_motion.device),
                    logvar=torch.zeros(pre_motion.shape[1], self.nz).to(
                        pre_motion.device
                    ),
                )
            else:
                data[prior_key] = Categorical(
                    logits=torch.zeros(pre_motion.shape[1], self.nz).to(
                        pre_motion.device
                    )
                )

        # sample latent code from the distribution
        if z is None:
            # use latent code z from posterior for training
            if mode == 'train':
                z = data["q_z_samp"]    # N x B x 32

            # use latent code z from posterior for evaluating the reconstruction loss
            elif mode == "recon":
                z = data["q_z_dist"].mode()   # NB x 32  
                z = z.view(agent_num, bs, z.size(-1))  # N x B x 32

            # use latent code z from the prior for inference
            elif mode == "infer":
                # dist =  data["p_z_dist_infer"] if "p_z_dist_infer" in data else data["q_z_dist_infer"]
                # z = dist.sample()  # NBS x 32
                # import pdb
                # pdb.set_trace()

                dist = data["q_z_dist"] if data["q_z_dist"] is not None else data["p_z_dist"]
                if self.z_type=="gaussian":
                    if predict:
                        z = dist.pseudo_sample(sample_num)
                    else:
                        z = data["p_z_dist_infer"].sample()
                    D = z.shape[-1]
                    samples = z.reshape(agent_num,bs,-1,D).permute(1,0,2,3)
                    mu = dist.mu.reshape(agent_num,bs,-1,D).permute(1,0,2,3)[:,:,0]
                    sigma = dist.sigma.reshape(agent_num,bs,-1,D).permute(1,0,2,3)[:,:,0]
                    data["prob"] = self.pseudo_sample_prob(samples,mu,sigma,data["agent_avail"])
                elif self.z_type=="discrete":
                    if predict:
                        z = dist.pseudo_sample(sample_num).contiguous()
                    else:
                        z = dist.rsample(sample_num).contiguous()
                    D = z.shape[-1]
                    idx = z.argmax(dim=-1)
                    prob_sample = torch.gather(dist.probs,-1,idx)
                    prob_sample = prob_sample.reshape(agent_num,bs,-1).prod(0)
                    prob_sample = prob_sample/prob_sample.sum(-1,keepdim=True)
                    data["prob"] = prob_sample

                z = z.view(agent_num, bs*sample_num, z.size(-1))  # N x BS x 32
            else:
                raise ValueError("Unknown Mode!")
    

        # trajectory decoding
        if autoregress:
            self.decode_traj_ar(
                data,
                mode,
                context,
                input_dict,
                z,
                sample_num,
                need_weights=need_weights,
                cond_idx=cond_idx
            )
            # self.decode_traj_ar_orig(
            #     data,
            #     mode,
            #     context,
            #     pre_motion,
            #     pre_vel,
            #     pre_motion_scene_norm,
            #     z,
            #     sample_num,
            #     need_weights=need_weights,
            # )
        else:
            self.decode_traj_batch(
                data,
                mode,
                context,
                input_dict,
                z,
                sample_num,
            )
    def pseudo_sample_prob(self,sample,mu,sigma,mask):
        """
        A simple K-means estimation to estimate the probability of samples
        """
        bs,Na,Ns,D = sample.shape
        device=sample.device
        Np = Ns*50
        particle = torch.randn([bs,Na,Np,D]).to(device)*sigma.unsqueeze(-2)+mu.unsqueeze(-2)
        dis = torch.linalg.norm(sample.unsqueeze(-2)-particle.unsqueeze(-3),dim=-1)
        dis = (dis*mask[...,None,None]).sum(1)
        idx = torch.argmin(dis,-2)
        flag = (idx.unsqueeze(1)==torch.arange(Ns).view(1,Ns,1).repeat_interleave(bs,0).to(device))
        prob = flag.sum(-1)/Np
        return prob


""" AgentFormer """
class AgentFormer(nn.Module):
    def __init__(self, algo_config, modality_shapes):
        super().__init__()

        self.algo_config = algo_config

        input_type = algo_config.input_type
        pred_type = algo_config.pred_type
        if type(input_type) == str:
            input_type = [input_type]
        fut_input_type = algo_config.fut_input_type
        dec_input_type = algo_config.dec_input_type

        self.use_map = algo_config.use_map
        self.rand_rot_scene = algo_config.rand_rot_scene
        self.discrete_rot = algo_config.discrete_rot
        self.map_global_rot = algo_config.map_global_rot
        self.ar_train = algo_config.ar_train
        self.max_train_agent = algo_config.max_train_agent
        self.loss_cfg = algo_config.loss_cfg
        self.param_annealers = nn.ModuleList()
        self.z_type = algo_config.z_type
        if self.z_type=="discrete":
            z_tau_annealer = ExpParamAnnealer(
                    algo_config.z_tau.start, algo_config.z_tau.finish, algo_config.z_tau.decay
                )
            self.param_annealers.append(z_tau_annealer)
            self.z_tau_annealer = z_tau_annealer
        self.ego_conditioning = algo_config.ego_conditioning
        self.step_time = algo_config.step_time
        self.dyn = dynamics.Unicycle()
        if "perturb" in algo_config and algo_config.perturb.enabled:
            self.N_pert = algo_config.perturb.N_pert
            theta = algo_config.perturb.OU.theta
            sigma = algo_config.perturb.OU.sigma
            scale = torch.tensor(algo_config.perturb.OU.scale)
            self.pert = DynOrnsteinUhlenbeckPerturbation(theta*torch.ones(self.dyn.udim),sigma*scale,self.dyn)
        else:
            self.N_pert = 0
            self.pert = None
        if "stage" in algo_config:
            assert algo_config.stage*algo_config.num_frames_per_stage<=algo_config.future_num_frames
            self.stage = algo_config.stage
            self.num_frames_per_stage = algo_config.num_frames_per_stage
        else:
            self.stage = 1
            self.num_frames_per_stage = algo_config.future_num_frames

        # save all computed variables
        self.data = dict()
        
        # map encoder
        if self.use_map:
            self.map_encoder = base_models.RasterizedMapEncoder(
                model_arch=algo_config.map_encoder.model_architecture,
                input_image_shape=algo_config.map_encoder.image_shape,
                feature_dim=algo_config.map_encoder.feature_dim,
                use_spatial_softmax=algo_config.map_encoder.spatial_softmax.enabled,
                spatial_softmax_kwargs=algo_config.map_encoder.spatial_softmax.kwargs,
            )

        # models
        self.context_encoder = ContextEncoder(algo_config)
        self.future_encoder = FutureEncoder(algo_config)
        self.future_decoder = FutureDecoder(algo_config)
        
    def set_data(self, batch, stage=0):

        device = batch["pre_motion_raw"].device
        self.data[stage] = batch
        self.data[stage]["step_time"] = self.step_time
        bs,Na = batch["pre_motion_raw"].shape[:2]
        self.data[stage]["pre_motion"] = (
            batch["pre_motion_raw"].to(device).transpose(0, 2).contiguous()
        )  # P x N x B x 2
        self.data[stage]["fut_motion"] = (
            batch["fut_motion_raw"].to(device).transpose(0, 2).contiguous()
        )  # F x N x B x 2

        # compute the origin of the current scene, i.e., the center
        # of the agents' location in the current frame
        self.data[stage]["scene_orig"] = torch.nanmean(
            self.data[stage]["pre_motion"][-1], dim=0
        )  # B x 2

        # normalize the scene with respect to the center location
        # optionally, also rotate the scene for augmentation
        if self.rand_rot_scene and self.training:
            # TODO: (Xinshuo) add visualization for double-checking here

            # below cannot be fixed in seed, causing reproducibility issue
            if self.discrete_rot:
                theta = torch.randint(high=24, size=(1,)).to(device) * (np.pi / 12)
            else:
                theta = torch.rand(1).to(device) * np.pi * 2  # [0, 2*pi], full circle

            for key in ["pre_motion", "fut_motion"]:
                (
                    self.data[stage][f"{key}"],
                    self.data[stage][f"{key}_scene_norm"],
                ) = rotation_2d_torch(
                    self.data[stage][key], theta, self.data[stage]["scene_orig"]
                )
            if self.data[stage]["heading"] is not None:
                self.data[stage]["heading"] += theta  # B x N
        else:
            theta = torch.zeros(1).to(device)

            # normalize per scene
            for key in ["pre_motion", "fut_motion"]:  # (F or P) x N x B x 2
                self.data[stage][f"{key}_scene_norm"] = (
                    self.data[stage][key] - self.data[stage]["scene_orig"]
                )

        # normalize pos per agent
        self.data[stage]["cur_motion"] = self.data[stage]["pre_motion"][[-1]]  # 1 x N x B x 2
        self.data[stage]["pre_motion_norm"] = (
            self.data[stage]["pre_motion"][:-1] - self.data[stage]["cur_motion"]  # P x N x B x 2
        )
        self.data[stage]["fut_motion_norm"] = (
            self.data[stage]["fut_motion"] - self.data[stage]["cur_motion"]
        )  # F x N x B x 2

        # vectorize heading
        if self.data[stage]["heading"] is not None:
            self.data[stage]["heading_vec"] = torch.stack(
                [torch.cos(self.data[stage]["heading"]), torch.sin(self.data[stage]["heading"])],
                dim=-1,
            ).transpose(0, 1)
            # N x B x 2
            self.data[stage]["pre_heading_vec"] = torch.stack(
                [torch.cos(self.data[stage]["pre_heading_raw"]), torch.sin(self.data[stage]["pre_heading_raw"])],
                dim=-1,
            ).transpose(0,2)
            # P x N x B x 2

            self.data[stage]["fut_heading_vec"] = torch.stack(
                [torch.cos(self.data[stage]["fut_heading_raw"]), torch.sin(self.data[stage]["fut_heading_raw"])],
                dim=-1,
            ).transpose(0,2)
            # F x N x B x 2

        # agent shuffling, default not shuffling
        if self.training and self.algo_config["agent_enc_shuffle"]:
            self.data[stage]["agent_enc_shuffle"] = torch.randperm(self.algo_config["max_agent_len"])[
                : self.data[stage]["agent_num"]
            ].to(device)
        else:
            self.data[stage]["agent_enc_shuffle"] = None

        # mask between pairwse agents, such as diable connection for a pair of agents
        # that are far away from each other, currently not used, i.e., assuming all connections
        conn_dist = self.algo_config.conn_dist
        cur_motion = self.data[stage]["cur_motion"][0]
        if conn_dist < 1000.0:
            threshold = conn_dist / self.algo_config.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D += D.T
            mask = torch.zeros_like(D)
            mask[D > threshold] = float("-inf")
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]]).to(device)
        self.data[stage]["agent_mask"] = mask  # N x N, all zeros now, i.e., fully-connected

    def step_annealer(self):
        for anl in self.param_annealers:
            anl.step()
    def convert_data(self,batch,cond_traj = None):
        data = defaultdict(lambda: None)
        ego_traj = torch.cat((batch["target_positions"][:,0],batch["target_yaws"][:,0]),-1)
        external_cond = True if cond_traj is not None else False
        if cond_traj is None:
            if self.pert is not None:
                # always perturb the ego trajectory
                
                ego_traj_tiled = ego_traj.repeat_interleave(self.N_pert,0)
                avail = batch["target_availabilities"][:,0].repeat_interleave(self.N_pert,0)
                pert_dict = self.pert.perturb(dict(target_positions=ego_traj_tiled[...,:2],\
                                target_yaws=ego_traj_tiled[...,2:],target_availabilities=avail,step_time=self.step_time))
                pert_ego_positions = pert_dict["target_positions"]
                pert_ego_trajectories = torch.cat((pert_dict["target_positions"],pert_dict["target_yaws"]),-1)

                cond_traj = torch.cat((ego_traj.unsqueeze(1),TensorUtils.reshape_dimensions_single(pert_ego_trajectories,0,1,[-1,self.N_pert])),1)
            else:
                cond_traj = ego_traj.unsqueeze(1)
            


    

        # TODO, should use more headings
        device = batch["history_positions"].device
        bs = batch["history_yaws"].shape[0]
        data["heading"] = batch["history_yaws"][:, :, -1, 0].to(device)    # B x N
        data["pre_heading_raw"] = batch["history_yaws"][..., 0].to(device)     # B x N x P
        data["fut_heading_full"] = batch["target_yaws"][...,0].to(device)       
        data["fut_heading_raw"] = data["fut_heading_full"][...,:self.num_frames_per_stage] # B x N x F
        traj_scale = self.algo_config.traj_scale
        data["traj_scale"] = traj_scale
        # AgentFormer uses the x/y inputs, i.e., the first two dimensions
        data["pre_motion_raw"] = (batch["history_positions"] / traj_scale).to(device)      # B x N x P x 2
        data["fut_motion_full"] = (batch["target_positions"] / traj_scale).to(device)  
        data["fut_motion_raw"] = (batch["target_positions"][:,:,:self.num_frames_per_stage] / traj_scale).to(device)       # B x N x F x 2

        data["pre_mask"] = batch["history_availabilities"].float().to(device)     # B x N x P                        # B x N x F x 2
        data["fut_mask_full"] = batch["target_availabilities"].float().to(device)     # B x N x F
        data["fut_mask"] = data["fut_mask_full"][...,:self.num_frames_per_stage]
        data["agent_avail"] = data["pre_mask"].any(-1).float()
        data["image"] = batch["image"]
        if cond_traj is not None and self.ego_conditioning:
            Ne = cond_traj.shape[1]
            if self.algo_config.UAC and not external_cond:
                Ne+=1
                cond_traj = torch.cat((torch.zeros_like(ego_traj).unsqueeze(1),cond_traj),1)
            for k in ["heading","pre_motion_raw","fut_motion_full","pre_mask","fut_mask_full","fut_mask","agent_avail","image","pre_heading_raw","fut_heading_raw","fut_heading_full"]:
                data[k] = data[k].repeat_interleave(Ne,0)
            if self.algo_config.UAC and not external_cond:
                data["fut_mask_full"][:bs] = False
                data["fut_mask"][:bs] = False
                
            fut_motion_full = data["fut_motion_full"]*traj_scale
            cond_traj_tiled = TensorUtils.join_dimensions(cond_traj,0,2)
            fut_motion_full[:,0] = cond_traj_tiled[...,:2]
            data["fut_heading_full"][:,0] = cond_traj_tiled[...,2]
            data["fut_heading_raw"] = data["fut_heading_full"][...,:self.num_frames_per_stage]
            data["fut_motion_full"] = fut_motion_full/ traj_scale
            data["fut_motion_raw"] = data["fut_motion_full"][:,:,:self.num_frames_per_stage]

        if self.ego_conditioning:
            data["cond_traj"] = cond_traj
        else:
            data["cond_traj"] = None
        data["pre_vel"] = DoubleIntegrator.calculate_vel(data["pre_motion_raw"],None,self.algo_config.step_time,data["pre_mask"].bool())
        data["pre_vel"] = data["pre_vel"].transpose(0,2).contiguous()
        data["fut_vel"] = DoubleIntegrator.calculate_vel(data["fut_motion_raw"],None,self.algo_config.step_time,data["fut_mask"].bool())     # F x N x B x 2
        data["fut_vel"] = data["fut_vel"].transpose(0,2).contiguous()

        return data

    def gen_data_stage(self,batch,pred_traj,stage):
        if stage==0:
            return batch
        else:
            data = defaultdict(lambda: None)
            device = pred_traj.device
            # fields that does not change
            for k in ["traj_scale","agent_enc_shuffle","fut_motion_full","fut_mask_full"]:
                data[k] = batch[k]
            traj_scale = self.algo_config.traj_scale
            bs,M,Na = pred_traj.shape[:3]
            data["heading"] = batch["heading"].repeat_interleave(M,0)    # (B*M) x N

            Ts = self.num_frames_per_stage
            P = self.algo_config.history_num_frames
            F = self.algo_config.future_num_frames
            if Ts<P:
                #left over from previous stage
                
                prev_stage_hist_pos = batch["pre_motion_raw"][:,:,Ts-P:].repeat_interleave(M,0)  # (B*M) x N x (P-Ts) x 2
                prev_stage_hist_yaw = batch["pre_heading_raw"][:,:,Ts-P:,2].repeat_interleave(M,0)  # (B*M) x N x (P-Ts)
                new_hist_pos = TensorUtils.join_dimensions(pred_traj[:,:,:,:Ts,:2],0,2)  # (B*M) x N x Ts x 2
                new_hist_yaw = TensorUtils.join_dimensions(pred_traj[:,:,:,:Ts,2],0,2)  # (B*M) x N x Ts

                data["pre_motion_raw"] = torch.cat((prev_stage_hist_pos,new_hist_pos),2)       # (B*M) x N x P x 2
                data["pre_heading_raw"] = torch.cat((prev_stage_hist_yaw,new_hist_yaw),2)       # (B*M) x N x P

                prev_stage_pre_mask = batch["pre_mask"][:,:,Ts-P:].repeat_interleave(M,0)    # (B*M) x N x (P-Ts)
                # since this is associated with the predicted trajectory, all entries is True except for dummy agents
                new_stage_pre_mask = batch["agent_avail"].unsqueeze(-1).repeat_interleave(M,0).repeat_interleave(Ts,-1) # (B*M) x N x Ts
                data["pre_mask"] = torch.cat((prev_stage_pre_mask,new_stage_pre_mask),-1)    # (B*M) x N x P
            else:
                data["pre_motion_raw"] = TensorUtils.join_dimensions(pred_traj[:,:,:,-P:,:2],0,2) # (B*M) x N x P x 2
                data["pre_heading_raw"] = TensorUtils.join_dimensions(pred_traj[:,:,:,-P:,2],0,2) # (B*M) x N x P
                data["pre_mask"] = batch["agent_avail"].unsqueeze(-1).repeat_interleave(M,0).repeat_interleave(P,-1)  # (B*M) x N x P
            # for future motion, pad the unknown future with 0

            data["fut_motion_raw"] = batch["fut_motion_full"][...,stage*Ts:(stage+1)*Ts,:].repeat_interleave(M**stage,0)          # (B*M) x N x Ts x 2
            data["fut_heading_raw"] = batch["fut_heading_full"][...,stage*Ts:(stage+1)*Ts].repeat_interleave(M**stage,0)          # (B*M) x N x Ts

            data["fut_mask"] = batch["fut_mask_full"][...,stage*Ts:(stage+1)*Ts].repeat_interleave(M**stage,0)

            data["agent_avail"] = batch["agent_avail"].repeat_interleave(M,0)

            data["pre_vel"] = DoubleIntegrator.calculate_vel(data["pre_motion_raw"],None,self.algo_config.step_time,data["pre_mask"].bool())
            data["pre_vel"] = data["pre_vel"].transpose(0,2).contiguous()
            data["fut_vel"] = DoubleIntegrator.calculate_vel(data["fut_motion_raw"],None,self.algo_config.step_time,data["fut_mask"].bool())     # F x N x B x 2
            data["fut_vel"] = data["fut_vel"].transpose(0,2).contiguous()
            data["map_enc"] = batch["map_enc"].repeat_interleave(M,1)   # N x (B*M) x D

            return data

    def forward(self,batch,sample_k=None, predict=False, **kwargs):
        cond_traj = kwargs["cond_traj"] if "cond_traj" in kwargs else None
            
        data0 = self.convert_data(batch,cond_traj=cond_traj)
        pred_traj = None
        pred_batch = dict()
        pred_batch["p_z_dist"] = dict()
        pred_batch["q_z_dist"] = dict()
        if self.ego_conditioning:
            cond_idx = [0]
        else:
            cond_idx = None
        data_stage = data0
        for stage in range(self.stage):
            data_stage = self.gen_data_stage(data_stage,pred_traj,stage)
            self.set_data(data_stage,stage)
            pred_data = self.run_model(stage,sample_k,predict=predict,cond_idx=cond_idx)
            pred_traj = pred_data["infer_dec_motion"]
            if "infer_dec_state" not in pred_data:
                yaws = torch.zeros_like(pred_traj[...,0:1])
            else:
                yaws = pred_data["infer_dec_state"][...,3:]

            pred_traj = torch.cat((pred_traj,yaws),-1)
            pred_batch["p_z_dist"][stage] = pred_data["p_z_dist"]
            pred_batch["q_z_dist"][stage] = pred_data["q_z_dist"]

        positions, state, var = self.batching_multistage_traj()
        positions = positions*self.algo_config.traj_scale
        NeB,numMode,Na,F = positions.shape[:4]
        bs = batch["history_positions"].shape[0]
        Ne = int(NeB/bs)
        if state is None:
            yaws = batch["history_yaws"][:,:,[-1]].repeat_interleave(F,2)
            
            yaws = yaws.unsqueeze(1).repeat_interleave(Ne,0).repeat_interleave(numMode,1)
            trajectories = torch.cat((positions,yaws),-1)
        else:
            trajectories= state[...,[0,1,3]]
        if "prob" not in self.data[0]:
            prob = torch.ones(trajectories.shape[:2]).to(trajectories.device)/trajectories.shape[1]
            prob = prob/prob.sum(-1,keepdim=True)
        else:
            
            M = int(numMode**(1/self.stage))
            prob = self.data[self.stage-1]["prob"].reshape(bs*Ne,*([M]*self.stage))

            for stage in range(self.stage-1):
                
                desired_shape = [bs*Ne]+[M]*(stage+1)+[1]*(self.stage-stage-1)
                prob = prob*TensorUtils.reshape_dimensions(self.data[stage]["prob"],0,2,desired_shape)
            prob = TensorUtils.join_dimensions(prob,1,self.stage+1)

        pred_except_dist = dict(trajectories=trajectories,p=prob,target_positions=data0["fut_motion_full"]*self.algo_config.traj_scale)
        pred_except_dist = TensorUtils.reshape_dimensions(pred_except_dist,0,1,[bs,Ne])
        pred_batch.update(pred_except_dist)
        pred_batch["cond_traj"]=data0["cond_traj"]
        agent_avail=self.data[0]["agent_avail"]
        agent_avail = agent_avail.reshape([bs,Ne,-1])[:,0]
        pred_batch["agent_avail"] = agent_avail
        pred_batch.update(self._traj_to_preds(pred_batch["trajectories"]))
        if var is not None:
            pred_batch["var"] = var
        if not predict:
            self.step_annealer()
        else:
            pred_batch = {k:v for k,v in pred_batch.items() if "dist" not in k}
        return pred_batch

    def batching_multistage_traj(self):
        if "infer_dec_motion" in self.data[0]:
            infer_traj = list()
            bs,M = self.data[0]["infer_dec_motion"].shape[:2]
            for stage in range(self.stage):
                traj_i = self.data[stage]["infer_dec_motion"].repeat_interleave((M**(self.stage-stage-1)),0)
                traj_i = traj_i.reshape(bs,M**self.stage,*traj_i.shape[2:])
                infer_traj.append(traj_i)
            infer_traj = torch.cat(infer_traj,-2)
        else:
            infer_traj = None
        if "infer_dec_state" in self.data[0]:
            infer_state = list()
            bs,M = self.data[0]["infer_dec_state"].shape[:2]
            for stage in range(self.stage):
                state_i = self.data[stage]["infer_dec_state"].repeat_interleave((M**(self.stage-stage-1)),0)
                state_i = state_i.reshape(bs,M**self.stage,*state_i.shape[2:])
                infer_state.append(state_i)
            infer_state = torch.cat(infer_state,-2)
        else:
            infer_state = None
        if "infer_var" in self.data[0]:
            infer_var = list()
            bs,M = self.data[0]["infer_var"].shape[:2]
            for stage in range(self.stage):
                var_i = self.data[stage]["infer_var"].repeat_interleave((M**(self.stage-stage-1)),0)
                var_i = var_i.reshape(bs,M**self.stage,*var_i.shape[2:])
                infer_var.append(var_i)
            infer_var = torch.cat(infer_var,-2)
        else:
            infer_state = None
            infer_var = None
            
        return infer_traj, infer_state, infer_var

    def sample(self,batch, sample_k):
        return self.forward(batch,sample_k)


    def run_model(self,stage,sample_k=None,predict=False,cond_idx=None):
        if self.use_map and self.data[stage]['map_enc'] is None:
            image = self.data[0]["image"]
            #hack
            image = image[...,-3:,:,:]
            bs,Na = image.shape[:2]
            map_enc = self.map_encoder(TensorUtils.join_dimensions(image,0,2))
            map_enc = map_enc.reshape(bs,Na,*map_enc.shape[1:])
            self.data[stage]["map_enc"] = map_enc.transpose(0,1)
        self.context_encoder(self.data[stage])
        if not predict:
            self.future_encoder(self.data[stage])
        # self.future_decoder(self.data[stage], mode='train', autoregress=self.ar_train)

        if sample_k is None:
            self.inference(sample_num=self.algo_config.sample_k,stage = stage,cond_idx=cond_idx,predict=predict)
        else:
            self.inference(sample_num=sample_k,stage = stage,cond_idx=cond_idx,predict=predict)

        # self.data[stage]["cond_traj"] = None
        return self.data[stage]

    def inference(self, mode='infer', sample_num=20, need_weights=False, stage = 0,cond_idx=None,predict=False):
        if self.use_map and self.data[stage]['map_enc'] is None:
            image = self.data[0]["image"]
            bs,Na = image.shape[:2]
            map_enc = self.map_encoder(TensorUtils.join_dimensions(image,0,2))
            map_enc = map_enc.reshape(bs,Na,*map_enc.shape[1:])
            self.data[stage]["map_enc"] = map_enc.transpose(0,1)
        if self.data[stage]['context_enc'] is None:
            self.context_encoder(self.data[stage])
        if mode == 'recon':
            sample_num = 1
            self.future_encoder(self.data[stage],temp=self.z_tau_annealer.val())
        
        if self.z_type=="gaussian":
            temp = None
        else:
            temp = 0.0001 if predict else self.z_tau_annealer.val()
            # raise Exception("one of p and q need to exist")
        
        self.future_decoder(self.data[stage], mode=mode, sample_num=sample_num, autoregress=True, need_weights=need_weights,cond_idx=cond_idx,temp=temp,predict=predict)
        return self.data[stage][f'{mode}_dec_motion'], self.data
    
    def _traj_to_preds(self, traj):
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        return {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws}
        }

    def compute_losses(self,pred_batch, data_batch):
        device = pred_batch["trajectories"]
        bs,Ne,numMode,Na = pred_batch["trajectories"].shape[:4]
        pred_batch["trajectories"] = pred_batch["trajectories"].nan_to_num(0)
        M = int(numMode**(1/self.stage))
        kl_loss = 0.0
        for stage in range(self.stage):
            kl_loss += pred_batch['q_z_dist'][stage].kl(pred_batch['p_z_dist'][stage]).nan_to_num(0).sum(-1).mean()

            
        kl_loss = kl_loss.clamp_min_(self.algo_config.loss_cfg.kld.min_clip)

        

        if self.algo_config.UAC:
            traj_pred_tiled = TensorUtils.join_dimensions(pred_batch["trajectories"][:,1:],0,2)
            traj_pred_tiled2 = TensorUtils.join_dimensions(traj_pred_tiled,0,2)
            Uncond_pred = pred_batch["trajectories"][:,0]
            cond_traj = pred_batch["cond_traj"][:,1:]
            var = pred_batch["var"][bs:]
            Ne = Ne-1
        else:
            traj_pred_tiled = TensorUtils.join_dimensions(pred_batch["trajectories"],0,2)
            traj_pred_tiled2 = TensorUtils.join_dimensions(traj_pred_tiled,0,2)
            cond_traj = pred_batch["cond_traj"]
        if Ne>1:
            fut_mask = data_batch["target_availabilities"].repeat_interleave(Ne,0)
        else:
            fut_mask = data_batch["target_availabilities"]
        if self.algo_config.UAC:
            pred_loss, goal_loss = MultiModal_trajectory_loss(
                predictions=pred_batch["trajectories"][:,0,...,:2], 
                targets=pred_batch["target_positions"][:,0], 
                availabilities=data_batch["target_availabilities"], 
                prob=pred_batch["p"][:,0], 
                calc_goal_reach=False,
            )
            flat_pred  =traj_pred_tiled[...,:2].reshape(-1,2)
            dev_loss = NLL_GMM_loss(flat_pred, 
                                    Uncond_pred[...,:2].repeat_interleave(Ne,0).reshape(-1,1,2),
                                    var.reshape(-1,1,2), 
                                    torch.ones([flat_pred.shape[0],1],device=var.device),
                                    detach=False, 
                                    mode="sum")
            dev_loss = torch.clip(dev_loss,min = 0)
        else:
            pred_loss, goal_loss = MultiModal_trajectory_loss(
                predictions=traj_pred_tiled[...,:2], 
                targets=TensorUtils.join_dimensions(pred_batch["target_positions"],0,2), 
                availabilities=fut_mask, 
                prob=TensorUtils.join_dimensions(pred_batch["p"],0,2), 
                calc_goal_reach=False,
            )
            dev_loss = torch.zeros(1).to(device)
        extent = data_batch["extent"][...,:2]
        div_score = diversity_score(traj_pred_tiled[...,:2],fut_mask.unsqueeze(1).repeat_interleave(numMode,1).any(1))
        # cond_extent = extent[torch.arange(bs),pred_batch["cond_idx"]]
        if pred_batch["cond_traj"] is not None:
            if "EC_coll_loss" in pred_batch:
                EC_coll_loss = pred_batch["EC_coll_loss"]
            else:
                EC_edges,type_mask = batch_utils().gen_EC_edges(
                    traj_pred_tiled2[:,1:],
                    cond_traj.reshape(bs*Ne,1,-1,3).repeat_interleave(numMode,0).repeat_interleave(Na-1,1),
                    extent[:,0].repeat_interleave(Ne*numMode,0),
                    extent[:,1:].repeat_interleave(Ne*numMode,0),
                    data_batch["type"][:,1:].repeat_interleave(Ne*numMode,0),
                    pred_batch["agent_avail"].repeat_interleave(Ne*numMode,0)[:,1:],
                )

                
                EC_edges = TensorUtils.reshape_dimensions(EC_edges,0,1,(bs,Ne,numMode))
                type_mask = TensorUtils.reshape_dimensions(type_mask,0,1,(bs,Ne,numMode))
                prob = pred_batch["p"][:,1:] if self.algo_config.UAC else pred_batch["p"]
                EC_coll_loss = collision_loss_masked(EC_edges,type_mask,weight=prob.reshape(bs,Ne,-1).unsqueeze(-1))
                if not isinstance(EC_coll_loss,torch.Tensor):
                    EC_coll_loss = torch.tensor(EC_coll_loss).to(device)
        else:
            EC_coll_loss = torch.tensor(0.0).to(device)
        
        # compute collision loss
        
        # pred_edges = batch_utils().generate_edges(
        #     pred_batch["agent_avail"].repeat_interleave(numMode*Ne,0),
        #     extent.repeat_interleave(Ne*numMode,0),
        #     traj_pred_tiled2[...,:2],
        #     traj_pred_tiled2[...,2:]
        # )
        
        # coll_loss = collision_loss(pred_edges=pred_edges)
        # if not isinstance(coll_loss,torch.Tensor):
        #     coll_loss = torch.tensor(coll_loss).to(device)
        
        losses = OrderedDict(
            prediction_loss=pred_loss,
            kl_loss = kl_loss,
            # collision_loss=coll_loss,
            EC_collision_loss = EC_coll_loss,
            diversity_loss = -div_score,
            # deviation_loss = dev_loss,
        )

        # if self.algo_config.input_weight_scaling is not None and "controls" in pred_batch:
        #     input_weight_scaling = torch.tensor(self.algo_config.input_weight_scaling).to(pred_batch["controls"].device)
        #     losses["input_loss"] = torch.mean(pred_batch["controls"] ** 2 *pred_batch["mask"][...,None]*input_weight_scaling)


        return losses



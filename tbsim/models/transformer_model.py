from logging import raiseExceptions
import numpy as np


from numpy.lib.function_base import flip
from tbsim.configs.base import AlgoConfig
import torch
import math
import copy
from typing import Dict
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tbsim.dynamics import Unicycle, DoubleIntegrator
from tbsim.dynamics.base import DynType
from tbsim.models.cnn_roi_encoder import (
    CNNROIMapEncoder,
    generate_ROIs,
    Indexing_ROI_result,
    rasterized_ROI_align,
    obtain_map_enc,
)
from tbsim.utils.tensor_utils import round_2pi
from tbsim.utils.geometry_utils import (
    VEH_VEH_collision,
    VEH_PED_collision,
    PED_VEH_collision,
    PED_PED_collision,
)
import tbsim.utils.l5_utils as L5Utils
import tbsim.utils.loss_utils as LossUtils
import tbsim.utils.tensor_utils as TensorUtils

from tbsim.models.Transformer import (
    make_transformer_model,
    simplelinear,
    subsequent_mask,
)
from tbsim.models.GAN_regularizer import pred2obs, pred2obs_static


class TransformerModel(nn.Module):
    def __init__(
            self,
            algo_config,
    ):
        super(TransformerModel, self).__init__()
        self.step_time = algo_config.step_time
        self.algo_config = algo_config
        self.calc_likelihood = algo_config.calc_likelihood
        self.M = algo_config.M
        self.goal_conditioned = algo_config.goal_conditioned

        self.register_buffer(
            "weights_scaling", torch.tensor(
                algo_config.weights.weights_scaling)
        )

        self.criterion = nn.MSELoss(reduction="none")
        self.map_enc_mode = algo_config.map_enc_mode
        "unicycle for vehicles and double integrators for pedestrians"
        self.dyn_list = {
            DynType.UNICYCLE: Unicycle(
                "vehicle", vbound=[algo_config.vmin, algo_config.vmax]
            ),
            DynType.DI: DoubleIntegrator(
                "pedestrian",
                abound=np.array([[-3.0, 3.0], [-3.0, 3.0]]),
                vbound=np.array([[-5.0, 5.0], [-5.0, 5.0]]),
            ),
        }
        if algo_config.calc_collision:
            self.col_funs = {
                "VV": VEH_VEH_collision,
                "VP": VEH_PED_collision,
                "PV": PED_VEH_collision,
                "PP": PED_PED_collision,
            }

        self.training_num = 0
        self.training_num_N = algo_config.training_num_N

        "src_dim:x,y,v,sin(yaw),cos(yaw)+16-dim type encoding"
        "tgt_dim:x,y,yaw"
        if algo_config.name == "TransformerGAN":
            self.use_GAN = True
            N_layer_enc_discr = algo_config.Discriminator.N_layer_enc
            self.GAN_static = algo_config.GAN_static
        else:
            self.use_GAN = False
            N_layer_enc_discr = None
            self.GAN_static = False

        if algo_config.vectorize_lane:
            src_dim = 21 + 3 * algo_config.points_per_lane * 5
        else:
            src_dim = 21
        self.Transformermodel, self.Discriminator = make_transformer_model(
            src_dim=src_dim,
            tgt_dim=4,
            out_dim=2,
            dyn_list=self.dyn_list.values(),
            N_t=algo_config.N_t,
            N_a=algo_config.N_a,
            d_model=algo_config.d_model,
            XY_pe_dim=algo_config.XY_pe_dim,
            temporal_pe_dim=algo_config.temporal_pe_dim,
            map_emb_dim=algo_config.map_emb_dim,
            d_ff=algo_config.d_ff,
            head=algo_config.head,
            dropout=algo_config.dropout,
            step_size=algo_config.XY_step_size,
            N_layer_enc=algo_config.N_layer_enc,
            N_layer_tgt_enc=algo_config.N_layer_tgt_enc,
            N_layer_tgt_dec=algo_config.N_layer_tgt_enc,
            M=self.M,
            use_GAN=self.use_GAN,
            GAN_static=self.GAN_static,
            N_layer_enc_discr=N_layer_enc_discr,
        )
        self.src_emb = nn.Linear(
            21,
            algo_config.d_model,
        ).cuda()

        "CNN for map encoding"
        self.CNNmodel = CNNROIMapEncoder(
            algo_config.CNN.map_channels,
            algo_config.CNN.hidden_channels,
            algo_config.CNN.ROI_outdim,
            algo_config.CNN.output_size,
            algo_config.CNN.kernel_size,
            algo_config.CNN.strides,
            algo_config.CNN.input_size,
        )

    @staticmethod
    def tgt_temporal_mask(p, tgt_mask):
        "use a binomial distribution with parameter p to mask out the first k steps of the tgt"
        nbatches = tgt_mask.size(0)
        T = tgt_mask.size(2)
        mask_hint = torch.ones_like(tgt_mask)
        sample = np.random.binomial(T, p, nbatches)
        for i in range(nbatches):
            mask_hint[i, :, sample[i]:] = 0
        return mask_hint

    def integrate_forward(self, x0, action, dyn_type):
        """
        Integrate the state forward with initial state x0, action u
        Args:
            x0 (Torch.tensor): state tensor of size [B,Num_agent,1,4]
            action (Torch.tensor): action tensor of size [B,Num_agent,T,2]
            dyn_type (Torch.tensor(dtype=int)): [description]
        Returns:
            state tensor of size [B,Num_agent,T,4]
        """
        T = action.size(-2)
        x = [x0.squeeze(-2)] + [None] * T
        veh_mask = (dyn_type == DynType.UNICYCLE).view(*dyn_type.shape, 1)
        ped_mask = (dyn_type == DynType.DI).view(*dyn_type.shape, 1)
        if action.ndim == 5:
            veh_mask = veh_mask.unsqueeze(1)
            ped_mask = ped_mask.unsqueeze(1)
        for t in range(T):
            x[t + 1] = (
                self.dyn_list[DynType.UNICYCLE].step(
                    x[t], action[..., t, :], self.step_time
                )
                * veh_mask
                + self.dyn_list[DynType.DI].step(
                    x[t], action[..., t, :], self.step_time
                )
                * ped_mask
            )

        x = torch.stack(x[1:], dim=-2)
        pos = self.dyn_list[DynType.UNICYCLE].state2pos(x) * veh_mask.unsqueeze(
            -1
        ) + self.dyn_list[DynType.DI].state2pos(x) * ped_mask.unsqueeze(-1)
        yaw = self.dyn_list[DynType.UNICYCLE].state2yaw(x) * veh_mask.unsqueeze(
            -1
        ) + self.dyn_list[DynType.DI].state2yaw(x) * ped_mask.unsqueeze(-1)
        return x, pos, yaw

    def forward(
            self, data_batch: Dict[str, torch.Tensor], batch_idx: int = None, plan: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        (
            src,
            dyn_type,
            src_state,
            src_pos,
            src_yaw,
            src_world_yaw,
            src_vel,
            raw_type,
            src_mask,
            src_lanes,
            extents,
            tgt_pos_yaw,
            tgt_mask,
            curr_state,
        ) = L5Utils.batch_to_vectorized_feature(
            data_batch, self.dyn_list, self.step_time, self.algo_config,
        )

        # Encode map
        map_emb = obtain_map_enc(
            data_batch["image"],
            self.CNNmodel,
            src_pos,
            src_yaw,
            data_batch["raster_from_agent"],
            src_mask,
            torch.tensor(self.algo_config.CNN.patch_size).to(src_pos.device),
            self.algo_config.CNN.output_size,
            self.map_enc_mode,
        )
        tgt_mask_agent = (
            tgt_mask.any(dim=-1).unsqueeze(-1).repeat(1, 1, tgt_mask.size(-1))
        )
        # seq_mask = subsequent_mask(tgt_mask.size(-1)).to(tgt_pos_yaw.device)
        # tgt_mask_dec = tgt_mask_agent.unsqueeze(-1) * seq_mask.unsqueeze(0)
        tgt_dec = torch.zeros([*tgt_pos_yaw.shape[:-1], 4]
                              ).to(tgt_pos_yaw.device)
        if self.goal_conditioned:
            if plan is not None:
                goal_pos = plan["positions"]
                goal_yaw = plan["yaws"]
                goal_mask = plan['availabilities']

            else:
                goal_pos, goal_yaw, goal_mask = L5Utils.obtain_goal_state(
                    tgt_pos_yaw, tgt_mask)
            goal_pos_rel = goal_pos - \
                curr_state[..., 0:2]*goal_mask.unsqueeze(-1)
            goal_yaw_rel = goal_yaw-curr_state[..., 3:]*goal_mask.unsqueeze(-1)
            T = tgt_pos_yaw.shape[-2]

            if goal_mask.shape[-1] >= T:
                tgt_dec = torch.cat([goal_pos_rel[:, :, :T], torch.cos(
                    goal_yaw_rel[:, :, :T]), torch.sin(goal_yaw_rel[:, :, :T])], dim=-1)
                goal_pos = goal_pos[:, :, :T]
                goal_yaw = goal_yaw[:, :, :T]
                goal_mask = goal_mask[:, :, :T]
            else:
                tgt_dec[:, :, :goal_mask.size(2)] = torch.cat(
                    [goal_pos_rel, torch.cos(goal_yaw_rel), torch.sin(goal_yaw_rel)], dim=-1)
                goal_pos = TensorUtils.pad_sequence_single(
                    goal_pos, [0, T-goal_mask.shape[-1]], batched=True, pad_dim=1)
                goal_yaw = TensorUtils.pad_sequence_single(
                    goal_yaw, [0, T-goal_mask.shape[-1]], batched=True, pad_dim=1)
                goal_mask = TensorUtils.pad_sequence_single(
                    goal_mask, [0, T-goal_mask.shape[-1]], batched=True, pad_dim=1)

            tgt_mask_dec = goal_mask.unsqueeze(-2).repeat(1,
                                                          1, tgt_mask.size(-1), 1)
        else:
            tgt_mask_dec = torch.zeros_like(
                tgt_mask).unsqueeze(-2).repeat(1, 1, tgt_mask.size(-1), 1)

        out, prob = self.Transformermodel.forward(
            src,
            tgt_dec,
            src_mask,
            tgt_mask_dec,
            tgt_mask_agent,
            dyn_type,
            map_emb,
        )

        u_pred = self.Transformermodel.generator(out)

        if self.M > 1:
            curr_state = curr_state.unsqueeze(1).repeat(1, self.M, 1, 1, 1)

        x_pred, pos_pred, yaw_pred = self.integrate_forward(
            curr_state, u_pred, dyn_type
        )
        lane_mask = (data_batch["image"][:, self.algo_config.CNN.lane_channel] < 1.0).type(
            torch.float)

        lane_flags = rasterized_ROI_align(
            lane_mask,
            pos_pred,
            yaw_pred,
            data_batch["raster_from_agent"],
            tgt_mask_agent,
            extents.type(torch.float)*self.algo_config.CNN.veh_patch_scale,
            self.algo_config.CNN.veh_ROI_outdim,
        )
        if self.M > 1:
            max_idx = torch.max(prob, dim=-1)[1]
            ego_pred_positions = pos_pred[torch.arange(
                0, pos_pred.size(0)), max_idx, 0]
            ego_pred_yaws = yaw_pred[torch.arange(
                0, pos_pred.size(0)), max_idx, 0]
        else:
            ego_pred_positions = pos_pred[:, 0]
            ego_pred_yaws = yaw_pred[:, 0]
        out_dict = {
            "predictions": {
                "positions": ego_pred_positions,
                "yaws": ego_pred_yaws,
            },
            "scene_predictions": {
                "positions": pos_pred,
                "yaws": yaw_pred,
                "prob": prob,
                "raw_outputs": x_pred,
                "lane_flags": lane_flags,
            },
        }

        if self.algo_config.calc_collision:
            out_dict["scene_predictions"]["edges"] = L5Utils.generate_edges(
                raw_type, extents, pos_pred, yaw_pred
            )

        if self.calc_likelihood:

            if self.GAN_static:
                src_noisy, _, __class__ = L5Utils.raw2feature(
                    src_pos,
                    src_vel,
                    src_yaw,
                    raw_type,
                    src_mask,
                    torch.zeros_like(
                        src_lanes) if src_lanes is not None else None,
                    add_noise=True,
                )
                src_rel = src_noisy[:, :, -1:].clone()
                src_rel[..., 0:2] -= (
                    src_noisy[:, 0:1, -1:, 0:2] * src_mask[:, :, -1:, None]
                )
                if map_emb.ndim == 4:
                    likelihood = self.Discriminator(
                        src_rel,
                        src_mask[:, :, -1:],
                        dyn_type,
                        map_emb[:, :, -1:],
                    ).view(src.shape[0], -1)
                else:
                    likelihood = self.Discriminator(
                        src_rel,
                        src_mask[:, :, -1:],
                        dyn_type,
                        map_emb.unsqueeze(-2),
                    ).view(src.shape[0], -1)
            else:
                likelihood = self.Discriminator(src, src_mask, dyn_type, map_emb).view(
                    src.shape[0], -1
                )
            if self.GAN_static:

                src_new, src_mask_new, map_emb_new = pred2obs_static(
                    self.dyn_list,
                    self.step_time,
                    data_batch,
                    pos_pred,
                    yaw_pred,
                    tgt_mask_agent,
                    raw_type,
                    torch.zeros_like(
                        src_lanes) if src_lanes is not None else None,
                    self.CNNmodel,
                    self.algo_config,
                    self.M,
                )
            else:
                src_new, src_mask_new, map_emb_new = pred2obs(
                    self.dyn_list,
                    self.step_time,
                    data_batch,
                    pos_pred,
                    yaw_pred,
                    tgt_mask_agent,
                    raw_type,
                    torch.zeros_like(
                        src_lanes) if src_lanes is not None else None,
                    self.CNNmodel,
                    self.algo_config,
                    self.algo_config.f_steps,
                    self.M,
                )
            if self.M == 1:
                src_new_rel = src_new.clone()
                src_new_rel[..., 0:2] -= src_new[
                    :, 0:1, :, 0:2
                ] * src_mask_new.unsqueeze(-1)
                likelihood_new = self.Discriminator(
                    src_new_rel, src_mask_new, dyn_type, map_emb_new
                ).view(src.shape[0], -1)
            else:
                src_new_rel = src_new.clone()
                src_new_rel[..., 0:2] -= src_new[
                    :, :, 0:1, :, 0:2
                ] * src_mask_new.unsqueeze(-1)
                likelihood_new = list()
                for i in range(self.M):
                    likelihood_new.append(
                        self.Discriminator(
                            src_new_rel[:, i],
                            src_mask_new[:, i],
                            dyn_type,
                            map_emb_new[:, i],
                        )
                    )
                likelihood_new = torch.stack(likelihood_new, dim=1).view(
                    src.shape[0], self.M, -1
                )
            out_dict["scene_predictions"]["likelihood_new"] = likelihood_new
            out_dict["scene_predictions"]["likelihood"] = likelihood
        if self.goal_conditioned:
            out_dict["scene_predictions"]["goal_pos"] = goal_pos
            out_dict["scene_predictions"]["goal_yaw"] = goal_yaw
            out_dict["scene_predictions"]["goal_mask"] = goal_mask
        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        if self.criterion is None:
            raise NotImplementedError("Loss function is undefined.")

        def yaw_crit(pred, target): return round_2pi(pred - target) ** 2

        ego_weights = data_batch["target_availabilities"] * \
            self.algo_config.weights.ego_weight

        all_other_types = data_batch["all_other_agents_types"]
        all_other_weights = (
            data_batch["all_other_agents_future_availability"] *
            self.algo_config.weights.all_other_weight
        )
        type_mask = ((all_other_types >= 3) & (
            all_other_types <= 13)).unsqueeze(-1)

        weights = torch.cat(
            (ego_weights.unsqueeze(1), all_other_weights * type_mask),
            dim=1,
        )
        eta = self.algo_config.temporal_bias
        T = pred_batch["predictions"]["yaws"].shape[-2]
        temporal_weight = (
            (1 - eta + torch.arange(T) / (T - 1) * 2 * eta)
            .view(1, 1, T)
            .to(weights.device)
        )
        weights = weights * temporal_weight
        mask = torch.cat(
            (
                data_batch["target_availabilities"].unsqueeze(1),
                data_batch["all_other_agents_future_availability"] * type_mask,
            ),
            dim=1,
        )
        weights_scaling = torch.tensor(
            self.algo_config.weights.weights_scaling).to(mask.device)
        instance_count = torch.sum(mask)

        scene_target_pos = torch.cat(
            (
                data_batch["target_positions"].unsqueeze(1),
                data_batch["all_other_agents_future_positions"],
            ),
            dim=1,
        )
        scene_target_yaw = torch.cat(
            (
                data_batch["target_yaws"].unsqueeze(1),
                data_batch["all_other_agents_future_yaws"],
            ),
            dim=1,
        )
        loss = 0
        if self.M == 1:
            loss += LossUtils.weighted_trajectory_loss(
                pred_batch["scene_predictions"]["positions"],
                scene_target_pos,
                weights,
                instance_count,
                weights_scaling[:2],
                crit=self.criterion,
            )

            loss += LossUtils.weighted_trajectory_loss(
                pred_batch["scene_predictions"]["yaws"] * mask.unsqueeze(-1),
                scene_target_yaw,
                weights,
                instance_count,
                weights_scaling[2:],
                crit=yaw_crit,
            )

        else:
            loss += LossUtils.weighted_multimodal_trajectory_loss(
                pred_batch["scene_predictions"]["positions"],
                scene_target_pos,
                weights,
                pred_batch["scene_predictions"]["prob"],
                instance_count,
                weights_scaling[:2],
                crit=self.criterion,
            )
            loss += LossUtils.weighted_multimodal_trajectory_loss(
                pred_batch["scene_predictions"]["yaws"],
                scene_target_yaw,
                weights,
                pred_batch["scene_predictions"]["prob"],
                instance_count,
                weights_scaling[2:],
                crit=yaw_crit,
            )

        if self.M == 1:
            lane_reg_loss = (
                LossUtils.lane_regularization_loss(
                    pred_batch["scene_predictions"]["lane_flags"],
                    weights,
                    instance_count,
                )
                * self.algo_config.weights.lane_regulation_weight
            )
        else:
            lane_reg_loss = (
                LossUtils.lane_regularization_loss(
                    pred_batch["scene_predictions"]["lane_flags"],
                    weights,
                    instance_count,
                    pred_batch["scene_predictions"]["prob"],
                )
                * self.algo_config.weights.lane_regulation_weight
            )

        losses = OrderedDict(
            prediction_loss=loss,
            lane_reg_loss=lane_reg_loss,
        )
        if self.goal_conditioned:
            goal_reaching_loss = 0
            goal_mask = pred_batch["scene_predictions"]["goal_mask"]
            goal_pos = pred_batch["scene_predictions"]["goal_pos"]
            goal_yaw = pred_batch["scene_predictions"]["goal_yaw"]
            goal_isntance_count = goal_mask.sum()
            if self.M == 1:
                goal_reaching_loss += LossUtils.weighted_trajectory_loss(
                    pred_batch["scene_predictions"]["positions"],
                    goal_pos,
                    goal_mask,
                    goal_isntance_count,
                    weights_scaling[:2],
                    crit=self.criterion,
                )

                goal_reaching_loss += LossUtils.weighted_trajectory_loss(
                    pred_batch["scene_predictions"]["yaws"] *
                    mask.unsqueeze(-1),
                    goal_yaw,
                    goal_mask,
                    goal_isntance_count,
                    weights_scaling[2:],
                    crit=yaw_crit,
                )
            else:
                goal_reaching_loss += LossUtils.weighted_multimodal_trajectory_loss(
                    pred_batch["scene_predictions"]["positions"],
                    goal_pos,
                    goal_mask,
                    pred_batch["scene_predictions"]["prob"],
                    goal_isntance_count,
                    weights_scaling[:2],
                    crit=self.criterion,
                )
                goal_reaching_loss += LossUtils.weighted_multimodal_trajectory_loss(
                    pred_batch["scene_predictions"]["yaws"],
                    scene_target_yaw,
                    goal_mask,
                    pred_batch["scene_predictions"]["prob"],
                    goal_isntance_count,
                    weights_scaling[2:],
                    crit=yaw_crit,
                )
            losses["goal_reaching_loss"] = goal_reaching_loss * \
                self.algo_config.weights.goal_reaching_weight
        if self.algo_config.calc_collision:

            coll_loss = LossUtils.collision_loss(
                pred_batch["scene_predictions"]["edges"], col_funcs=self.col_funs
            )
            losses["coll_loss"] = coll_loss * \
                self.algo_config.weights.collision_weight
        if self.algo_config.calc_likelihood:
            likelihood_loss = self.algo_config.weights.GAN_weight * \
                LossUtils.likelihood_loss(
                    pred_batch["scene_predictions"]["likelihood_new"])
            losses["likelihood_loss"] = likelihood_loss

        return losses

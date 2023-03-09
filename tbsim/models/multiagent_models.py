from typing import Dict, List
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tbsim.models.base_models as base_models
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.policies.common import Plan

from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import get_upright_box, transform_points_tensor, calc_distance_map
from tbsim.utils.loss_utils import (
    trajectory_loss,
    goal_reaching_loss,
    collision_loss,
    lane_regulation_loss,
)
from tbsim.models.roi_align import ROI_align, generate_ROIs, Indexing_ROI_result
from tbsim.models.cnn_roi_encoder import rasterized_ROI_align
from tbsim.models.Transformer import SimpleTransformer


class AgentAwareRasterizedModel(nn.Module):
    """Ego-centric model that is aware of other agents' future trajectories through auxiliary prediction task"""

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            global_feature_dim: int,
            agent_feature_dim: int,
            future_num_frames: int,
            roi_size: tuple,
            dynamics_type: str,
            dynamics_kwargs: dict,
            step_time: float,
            decoder_kwargs: dict = None,
            goal_conditional: bool = False,
            goal_feature_dim: int = 32,
            weights_scaling: tuple = (1.0, 1.0, 1.0),
            use_transformer=True,
            use_rotated_roi=True,
            history_conditioning=False,
            use_gan=False,
            roi_layer_key="layer4"
    ) -> None:

        nn.Module.__init__(self)

        self.map_encoder = base_models.RasterizeROIEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,  # [C, H, W]
            global_feature_dim=global_feature_dim,
            agent_feature_dim=agent_feature_dim,
            output_activation=nn.ReLU,
            use_rotated_roi=use_rotated_roi,
            roi_layer_key=roi_layer_key
        )
        self.use_rotated_roi = use_rotated_roi
        self.goal_conditional = goal_conditional
        self.history_conditioning = history_conditioning

        goal_dim = 0
        if self.goal_conditional:
            self.goal_encoder = base_models.MLP(
                input_dim=3,
                output_dim=goal_feature_dim,
                output_activation=nn.ReLU
            )
            goal_dim = goal_feature_dim

        hist_feat_dim = 0
        if history_conditioning:
            hist_feat_dim = 16
            self.history_encoder = base_models.RNNTrajectoryEncoder(
                trajectory_dim=3,
                rnn_hidden_size=100,
                mlp_layer_dims=(128, 128),
                feature_dim=hist_feat_dim
            )

        self.ego_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=agent_feature_dim + global_feature_dim + goal_dim + hist_feat_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )


        # other_dyn_type = None if disable_dynamics_for_other_agents else dynamics_type
        self.agents_decoder = base_models.MLPTrajectoryDecoder(
            feature_dim=agent_feature_dim + global_feature_dim + hist_feat_dim,
            state_dim=3,
            num_steps=future_num_frames,
            dynamics_type=dynamics_type,
            dynamics_kwargs=dynamics_kwargs,
            step_time=step_time,
            network_kwargs=decoder_kwargs
        )
        if use_transformer:
            self.transformer = SimpleTransformer(
                src_dim=agent_feature_dim + global_feature_dim+hist_feat_dim)
        else:
            self.transformer = None

        if use_gan:
            traj_enc_dim = 64
            self.traj_encoder = base_models.MLP(
                input_dim=2*future_num_frames, output_dim=traj_enc_dim, layer_dims=(64, 64))
            # TODO: make this part of the config
            self.gan_disc = base_models.MLP(input_dim=agent_feature_dim + global_feature_dim + goal_dim + traj_enc_dim,
                                            output_dim=1,
                                            layer_dims=(256, 128),
                                            output_activation=nn.Sigmoid)
        else:
            self.gan_disc = None

        assert len(roi_size) == 2
        self.roi_size = nn.Parameter(
            torch.Tensor([roi_size[0], roi_size[0], roi_size[1],
                         roi_size[1]]),  # [W1, W2, H1, H2]
            requires_grad=False
        )
        self.weights_scaling = nn.Parameter(
            torch.Tensor(weights_scaling), requires_grad=False)

    @staticmethod
    def get_ego_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 0])

    @staticmethod
    def get_agents_predictions(pred_batch):
        return TensorUtils.map_tensor(pred_batch, lambda x: x[:, 1:])

    def compute_metrics(self, pred_batch, data_batch):
        metrics = dict()

        # ego ADE & FDE
        ego_preds = self.get_ego_predictions(pred_batch)
        pos_preds = TensorUtils.to_numpy(ego_preds["predictions"]["positions"])

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, pos_preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, pos_preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        # agent ADE & FDE
        agents_preds = self.get_agents_predictions(pred_batch)
        pos_preds = TensorUtils.to_numpy(
            agents_preds["predictions"]["positions"])
        num_frames = pos_preds.shape[2]
        pos_preds = pos_preds.reshape(-1, num_frames, 2)
        all_targets = batch_utils().batch_to_target_all_agents(data_batch)
        gt = TensorUtils.to_numpy(
            all_targets["target_positions"][:, 1:]).reshape(-1, num_frames, 2)
        avail = TensorUtils.to_numpy(
            all_targets["target_availabilities"][:, 1:]).reshape(-1, num_frames)

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, pos_preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, pos_preds, avail
        )

        metrics["agents_ADE"] = np.mean(ade)
        metrics["agents_FDE"] = np.mean(fde)

        return metrics

    def _get_roi_boxes_upright(self, pos, trans_mat, patch_size):
        b, a = pos.shape[:2]
        curr_pos_raster = transform_points_tensor(pos, trans_mat.float())
        extents = torch.ones_like(curr_pos_raster) * patch_size  # [B, A, 2]
        rois_raster = get_upright_box(
            curr_pos_raster, extent=extents).reshape(b * a, 2, 2)
        rois_raster = torch.flatten(rois_raster, start_dim=1)  # [B * A, 4]

        roi_indices = torch.arange(0, b).unsqueeze(
            1).expand(-1, a).reshape(-1, 1).to(rois_raster.device)  # [B * A, 1]
        indexed_rois_raster = torch.cat(
            (roi_indices, rois_raster), dim=1)  # [B * A, 5]
        return indexed_rois_raster

    def _get_roi_boxes_rotated(self, pos, yaw, avails, trans_mat, patch_size):
        rois, indices = generate_ROIs(
            pos, yaw, trans_mat, avails, patch_size, mode="last")
        return rois, indices

    def _get_goal_states(self, data_batch, plan=None) -> torch.Tensor:
        if plan is None:
            all_targets = batch_utils().batch_to_target_all_agents(data_batch)
            target_traj = torch.cat(
                (all_targets["target_positions"], all_targets["target_yaws"]), dim=-1)
            goal_inds = batch_utils().get_last_available_index(
                all_targets["target_availabilities"])  # [B, A]
            goal_state = torch.gather(
                target_traj,  # [B, A, T, 3]
                dim=2,
                # [B, A, 1, 3]
                index=goal_inds[:, :, None,
                                None].expand(-1, -1, 1, target_traj.shape[-1])
            ).squeeze(2)  # -> [B, A, 3]
            return goal_state
        else:
            assert isinstance(plan, Plan)
            goal_inds = batch_utils().get_last_available_index(
                plan.availabilities)  # [B, A]
            goal_state = torch.gather(
                plan.trajectories,  # [B, T, 3]
                dim=1,
                # [B, 1, 3]
                index=goal_inds[:, None,
                                None].expand(-1, 1, plan.trajectories.shape[-1])
            ).squeeze(1)  # -> [B, 3]
            return goal_state

    def _get_lane_flags(self, data_batch, pred_yaws, pred_positions):
        lane_mask = batch_utils().get_drivable_region_map(
            data_batch["image"]).float()
        dis_map = calc_distance_map(lane_mask)
        target_mask = torch.cat([data_batch["target_availabilities"].unsqueeze(
            1), data_batch["all_other_agents_future_availability"]], 1)
        agent_mask = target_mask.any(
            dim=-1).unsqueeze(-1).repeat(1, 1, target_mask.size(-1))
        extents = torch.cat(
            (
                data_batch["extent"][..., :2].unsqueeze(1),
                torch.max(
                    data_batch["all_other_agents_history_extents"], dim=-2)[0],
            ),
            dim=1,
        )
        # pred_world_yaws = pred_yaws + data_batch['yaw'].reshape(-1,1,1,1)
        lane_flags = rasterized_ROI_align(dis_map,
                                      pred_positions,
                                      pred_yaws,
                                      data_batch["raster_from_agent"],
                                      agent_mask,
                                      extents.type(torch.float),
                                      3,
                                      ).squeeze(-1)
        return lane_flags

    def extract_features(self, data_batch, return_encoder_feats=False):
        image_batch = data_batch["image"]
        states_all = batch_utils().batch_to_raw_all_agents(
            data_batch, self.ego_decoder.step_time)
        b, a = states_all["history_positions"].shape[:2]

        # extract agent-wise features
        if self.use_rotated_roi:
            rois, indices = self._get_roi_boxes_rotated(
                pos=states_all["history_positions"],
                yaw=states_all["history_yaws"],
                avails=states_all["history_availabilities"],
                trans_mat=data_batch["raster_from_agent"],
                patch_size=self.roi_size
            )

            all_feats, _, global_feats, encoder_feats = self.map_encoder(
                image_batch, rois=rois)  # approximately B * A
            split_sizes = [len(l) for l in indices]
            all_feats_list = torch.split(all_feats, split_sizes)
            all_feats = Indexing_ROI_result(
                all_feats_list, indices, emb_size=(b, a, all_feats.shape[-1]))

        else:
            curr_pos_all = torch.cat((
                data_batch["history_positions"].unsqueeze(1),
                data_batch["all_other_agents_history_positions"],
            ), dim=1)[:, :, 0]  # histories are reversed
            rois = self._get_roi_boxes_upright(
                curr_pos_all,
                trans_mat=data_batch["raster_from_agent"],
                patch_size=self.roi_size[[0, 2]]
            )
            all_feats, _, global_feats, encoder_feats = self.map_encoder(
                image_batch, rois=rois)

        # tile global feature and concat w/ agent-wise features
        all_feats = all_feats.reshape(b, a, -1)
        all_feats = torch.cat(
            (all_feats, TensorUtils.unsqueeze_expand_at(global_feats, a, 1)), dim=-1)

        if self.history_conditioning:
            hist_traj = torch.cat((states_all["history_positions"], states_all["history_yaws"]), dim=-1)
            hist_feats = TensorUtils.time_distributed(hist_traj, self.history_encoder)
            all_feats = torch.cat((all_feats, hist_feats), dim=-1)

        # optionally pass information using transformer
        if self.transformer is not None:
            all_feats = self.transformer(
                all_feats,
                states_all["history_availabilities"][:, :, -1],
                states_all["history_positions"][:, :, -1]
            )

        if not return_encoder_feats:
            return all_feats
        else:
            return all_feats, encoder_feats

    def forward_prediction(self, all_feats, data_batch, plan=None):
        ego_feats = all_feats[:, [0]]
        agents_feats = all_feats[:, 1:]

        if self.goal_conditional:
            if plan is None:
                # optionally condition the ego prediction on a goal location
                goal_state = self._get_goal_states(data_batch)[:, [0]]
            else:
                goal_state = self._get_goal_states(data_batch, plan).unsqueeze(1)
            goal_feat = self.goal_encoder(goal_state)  # -> [B, 1, D]
            ego_feats = torch.cat((ego_feats, goal_feat), dim=-1)

        # for dynamics models
        if self.ego_decoder.dyn is not None:
            dyn_states = batch_utils().get_current_states_all_agents(
                data_batch,
                self.ego_decoder.step_time,
                dyn_type=self.ego_decoder.dyn.type()
            )
            ego_states = dyn_states[:, [0]]
            agents_states = dyn_states[:, 1:]
        else:
            ego_states = None
            agents_states = None

        # make predictions
        ego_preds = self.ego_decoder.forward(
            inputs=ego_feats, current_states=ego_states,predict=True)
        agents_preds = self.agents_decoder.forward(
            inputs=agents_feats, current_states=agents_states,predict=True)

        # summarize predictions
        all_preds = dict()
        for k in ego_preds:
            all_preds[k] = torch.cat((ego_preds[k], agents_preds[k]), dim=1)

        traj = all_preds["trajectories"]
        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]
        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
        }
        if self.ego_decoder.dyn is not None:
            out_dict["controls"] = all_preds["controls"]
        return out_dict

    def forward(self, data_batch: Dict[str, torch.Tensor], plan=None) -> Dict[str, torch.Tensor]:
        all_feats = self.extract_features(data_batch)
        pred_dict = self.forward_prediction(all_feats, data_batch, plan=plan)

        if self.gan_disc is not None:
            b = all_feats.shape[0]
            ego_feats = all_feats[:, 0]
            traj_enc_feat_GT = self.traj_encoder(
                data_batch["target_positions"][..., :2].reshape(b, -1).detach())
            likelihood_GT = self.gan_disc(
                torch.cat((ego_feats, traj_enc_feat_GT), -1))

            ego_preds = self.get_ego_predictions(pred_dict)
            traj_enc_feat_pred = self.traj_encoder(
                ego_preds["trajectories"][..., :2].reshape(b, -1))
            likelihood_pred = self.gan_disc(
                torch.cat((ego_feats, traj_enc_feat_pred), -1))
            pred_dict["likelihood_GT"] = likelihood_GT
            pred_dict["likelihood_pred"] = likelihood_pred
        return pred_dict

    def compute_losses(self, pred_batch, data_batch):
        all_targets = batch_utils().batch_to_target_all_agents(data_batch)
        target_traj = torch.cat(
            (all_targets["target_positions"], all_targets["target_yaws"]), dim=-1)
        pred_loss = trajectory_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        goal_loss = goal_reaching_loss(
            predictions=pred_batch["trajectories"],
            targets=target_traj,
            availabilities=all_targets["target_availabilities"],
            weights_scaling=self.weights_scaling
        )

        # compute collision loss
        # preds = TensorUtils.clone(pred_batch["predictions"])
        # pred_edges = batch_utils().get_edges_from_batch(
        #     data_batch=data_batch,
        #     all_predictions=preds
        # )

        # coll_loss = collision_loss(pred_edges=pred_edges)

        # target_mask = torch.cat(
        #     [
        #         data_batch["target_availabilities"].unsqueeze(1),
        #         data_batch["all_other_agents_future_availability"]
        #     ],
        #     dim=1
        # )
        # agent_mask = target_mask.any(dim=-1)
        # lane_reg_loss = lane_regulation_loss(pred_batch["lane_flags"],agent_mask)

        losses = OrderedDict(
            prediction_loss=pred_loss,
            goal_loss=goal_loss,
            # collision_loss=coll_loss
        )
        if self.gan_disc is not None:
            imitation_loss = F.binary_cross_entropy(
                pred_batch["likelihood_pred"], torch.ones_like(pred_batch["likelihood_pred"]))
            losses["GAN_loss"] = imitation_loss

        if "controls" in pred_batch:
            # regularize the magnitude of yaw control
            losses["yaw_reg_loss"] = torch.mean(
                pred_batch["controls"][..., 1] ** 2)

        return losses

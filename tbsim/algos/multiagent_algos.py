from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.multiagent_models import (
    AgentAwareRasterizedModel
)
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import Action, Plan, Trajectory
from tbsim.utils.loss_utils import discriminator_loss
from tbsim.utils.batch_utils import batch_utils
from tbsim.models.base_models import RasterizedMapUNet
from tbsim.algos.algos import SpatialPlanner
from tbsim.utils.geometry_utils import calc_distance_map
from tbsim.utils.planning_utils import ego_sample_planning
import tbsim.algos.algo_utils as AlgoUtils


class MATrafficModel(pl.LightningModule):
    """Prediction module for prediction-and-planning."""
    def __init__(self, algo_config, modality_shapes):
        super(MATrafficModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        self.model = AgentAwareRasterizedModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            global_feature_dim=algo_config.global_feature_dim,
            agent_feature_dim=algo_config.agent_feature_dim,
            roi_size=algo_config.context_size,
            future_num_frames=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            decoder_kwargs=algo_config.decoder,
            goal_conditional=algo_config.goal_conditional,
            goal_feature_dim=algo_config.goal_feature_dim,
            use_rotated_roi=algo_config.use_rotated_roi,
            use_transformer=algo_config.use_transformer,
            history_conditioning=algo_config.history_conditioning,
            roi_layer_key=algo_config.roi_layer_key,
            use_gan=algo_config.use_GAN
        )

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict, plan=None):
        return self.model(obs_dict, plan)

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.model.forward(batch)
        losses = self.model.compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self.model.compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.model.forward(batch)
        losses = TensorUtils.detach(self.model.compute_losses(pout, batch))
        metrics = self.model.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, obs_dict, **kwargs):
        """If using the model as a planner (setting subgoals)"""
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        avails = torch.ones(ego_preds["predictions"]["positions"].shape[:-1]).to(
            ego_preds["predictions"]["positions"].device)

        plan = Plan(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"],
            availabilities=avails
        )

        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        """If using the model as an actor (generating actions)"""
        # extract agent features from obs
        feats = self.model.extract_features(obs_dict)
        if "plan_samples" in kwargs:
            # if evaluating multiple plan samples per obs
            plan_samples = kwargs["plan_samples"]
            b, n = plan_samples.positions.shape[:2]
            # reuse features by tiling the feature tensors to the same size as plan samples
            feats_tiled = TensorUtils.repeat_by_expand_at(feats, repeats=n, dim=0)
            # flatten the sample dimension to the batch dimension
            plan_tiled = TensorUtils.join_dimensions(plan_samples.to_dict(), begin_axis=0, end_axis=2)
            plan_tiled = Plan.from_dict(plan_tiled)

            relevant_keys = ["curr_speed", "history_positions", "history_yaws", "extent","all_other_agents_history_extents",\
                "type","all_other_agents_types","all_other_agents_history_positions",\
                    "all_other_agents_history_yaws","history_availabilities",\
                        "all_other_agents_history_availability","all_other_agents_curr_speed"]
            relevant_obs = {k: obs_dict[k] for k in relevant_keys}
            obs_tiled = TensorUtils.repeat_by_expand_at(relevant_obs, repeats=n, dim=0)
            preds = self.model.forward_prediction(feats_tiled, obs_tiled, plan=plan_tiled)
        else:
            plan = kwargs.get("plan", None)
            preds = self.model.forward_prediction(feats, obs_dict, plan)

        ego_preds = self.model.get_ego_predictions(preds)
        action = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        return action, {}

    def get_prediction(self, obs_dict, **kwargs):
        """If using the model as a trajectory predictor (generating trajs for non-ego agents)"""
        # Hack: ego can be goal-conditional - feed a fake goal here since we only care about other agents
        dummy_plan = Plan(
            positions=torch.zeros(obs_dict["image"].shape[0], 1, 2).to(self.device),
            yaws=torch.zeros(obs_dict["image"].shape[0], 1, 1).to(self.device),
            availabilities=torch.zeros(obs_dict["image"].shape[0], 1).to(self.device)
        )
        plan = kwargs.get("plan", dummy_plan)
        preds = self(obs_dict, plan)
        agent_preds = self.model.get_agents_predictions(preds)
        agent_trajs = Trajectory(
            positions=agent_preds["predictions"]["positions"],
            yaws=agent_preds["predictions"]["yaws"]
        )
        return agent_trajs, {}


class MAGANTrafficModel(MATrafficModel):

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict, plan=None):
        return self.model(obs_dict, plan)

    def discriminator_loss(self, pred_batch):
        d_loss = discriminator_loss(
            pred_batch["likelihood_pred"], pred_batch["likelihood_GT"])
        self.log("train/discriminator_loss", d_loss)
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.model.forward(batch)
        if optimizer_idx == 0:
            losses = self.model.compute_losses(pout, batch)
            total_loss = 0.0
            for lk, l in losses.items():
                loss = l * self.algo_config.loss_weights[lk]
                self.log("train/losses_" + lk, loss)
                total_loss += loss

            metrics = self.model.compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m)

            return total_loss
        elif optimizer_idx == 1:
            d_loss = self.discriminator_loss(pout)

            for mk in ["likelihood_pred", "likelihood_GT"]:
                self.log("train/metrics_" + mk, pout[mk].mean())
            return d_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.model.forward(batch)
        losses = TensorUtils.detach(self.model.compute_losses(pout, batch))
        metrics = self.model.compute_metrics(pout, batch)
        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        gen_params = list()
        discr_params = list()
        for com_name, com in self.model.named_children():
            if com_name not in ["gan_disc", "traj_encoder"]:
                gen_params += list(com.parameters())
            else:
                discr_params += list(com.parameters())
        gen_optim_params = self.algo_config.optim_params.policy
        discr_optim_params = self.algo_config.optim_params.GAN
        gen_optim = optim.Adam(
            params=gen_params,
            lr=gen_optim_params["learning_rate"]["initial"],
            weight_decay=gen_optim_params["regularization"]["L2"],
        )
        discr_optim = optim.Adam(
            params=discr_params,
            lr=discr_optim_params["learning_rate"]["initial"],
            weight_decay=discr_optim_params["regularization"]["L2"],
        )

        return [gen_optim, discr_optim], []

    def get_plan(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        ego_preds = self.model.get_ego_predictions(preds)
        avails = torch.ones(ego_preds["predictions"]["positions"].shape[:-1]).to(
            ego_preds["predictions"]["positions"].device)

        plan = Plan(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"],
            availabilities=avails
        )

        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        if "plan" in kwargs:
            plan = kwargs["plan"]
        else:
            plan = None
        preds = self(obs_dict, plan)
        ego_preds = self.model.get_ego_predictions(preds)
        action = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        return action, {}

    def get_prediction(self, obs_dict, **kwargs):
        if "plan" in kwargs:
            plan = kwargs["plan"]
        else:
            plan = None
        preds = self(obs_dict, plan)
        agent_preds = self.model.get_agents_predictions(preds)
        agent_trajs = Trajectory(
            positions=agent_preds["predictions"]["positions"],
            yaws=agent_preds["predictions"]["yaws"]
        )
        return agent_trajs, {}


class HierarchicalAgentAwareModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(HierarchicalAgentAwareModel, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        assert not algo_config.use_GAN  # TODO: fix forward
        self.predictor = AgentAwareRasterizedModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            global_feature_dim=algo_config.global_feature_dim,
            agent_feature_dim=algo_config.agent_feature_dim,
            roi_size=algo_config.context_size,
            future_num_frames=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            decoder_kwargs=algo_config.decoder,
            goal_conditional=algo_config.goal_conditional,
            goal_feature_dim=algo_config.goal_feature_dim,
            use_rotated_roi=algo_config.use_rotated_roi,
            use_transformer=algo_config.use_transformer,
            history_conditioning=algo_config.history_conditioning,
            roi_layer_key=algo_config.roi_layer_key,
            use_gan=algo_config.use_GAN
        )

        self.planner = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residual, y_residual, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )
        self.planner.encoder_heads = None  # we are going to use the feature from the predictor

    @property
    def checkpoint_monitor_keys(self):
        return {
            "valLoss": "val/losses_prediction_loss"
        }

    def forward(self, obs_dict):
        all_feats, encoder_feats = self.predictor.extract_features(obs_dict, return_encoder_feats=True)
        pred_maps = self.planner.forward(None, encoder_feats=encoder_feats)
        planner_preds = SpatialPlanner.forward_prediction(pred_map=pred_maps, obs_dict=obs_dict)
        predictor_preds = self.predictor.forward_prediction(all_feats, data_batch=obs_dict, plan=None)
        return planner_preds, predictor_preds

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        planner_pout, predictor_pout = self.forward(batch)
        predictor_losses = self.predictor.compute_losses(predictor_pout, batch)
        planner_losses = SpatialPlanner.compute_losses(planner_pout, batch)
        losses = dict()
        losses.update(predictor_losses)
        losses.update(planner_losses)

        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        predictor_metrics = self.predictor.compute_metrics(predictor_pout, batch)
        planner_metrics = SpatialPlanner.compute_metrics(planner_pout, batch)

        metrics = dict()
        metrics.update(predictor_metrics)
        metrics.update(planner_metrics)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        planner_pout, predictor_pout = self.forward(batch)
        with torch.no_grad():
            predictor_losses = self.predictor.compute_losses(predictor_pout, batch)
            planner_losses = SpatialPlanner.compute_losses(planner_pout, batch)
        losses = dict()
        losses.update(predictor_losses)
        losses.update(planner_losses)

        with torch.no_grad():
            predictor_metrics = self.predictor.compute_metrics(predictor_pout, batch)
            planner_metrics = SpatialPlanner.compute_metrics(planner_pout, batch)

        metrics = dict()
        metrics.update(predictor_metrics)
        metrics.update(planner_metrics)

        return {"losses": losses, "metrics": metrics}

    def validation_epoch_end(self, outputs) -> None:
        for k in outputs[0]["losses"]:
            m = torch.stack([o["losses"][k] for o in outputs]).mean()
            self.log("val/losses_" + k, m)

        for k in outputs[0]["metrics"]:
            m = np.stack([o["metrics"][k] for o in outputs]).mean()
            self.log("val/metrics_" + k, m)

    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params["policy"]
        return optim.Adam(
            params=self.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

    def get_plan(self, preds):
        b, n = preds["predictions"]["positions"].shape[:2]
        plan_dict = dict(
            predictions=TensorUtils.unsqueeze(preds["predictions"], dim=1),  # [B, 1, num_sample...]
            availabilities=torch.ones(b, 1, n).to(self.device),  # [B, 1, num_sample]
        )
        # pad plans to the same size as the future trajectories
        n_steps_to_pad = self.algo_config.future_num_frames - 1
        plan_dict = TensorUtils.pad_sequence(plan_dict, padding=(n_steps_to_pad, 0), batched=True, pad_values=0.)
        plan_samples = Plan(
            positions=plan_dict["predictions"]["positions"].permute(0, 2, 1, 3),  # [B, num_sample, T, 2]
            yaws=plan_dict["predictions"]["yaws"].permute(0, 2, 1, 3),  # [B, num_sample, T, 1]
            availabilities=plan_dict["availabilities"].permute(0, 2, 1)  # [B, num_sample, T]
        )
        plan_info = dict(
            location_map=preds["location_map"],
            plan_samples=plan_samples,
            log_likelihood=preds["log_likelihood"]
        )

        return plan_samples, plan_info

    def get_action(self, obs_dict, **kwargs):
        """If using the model as an actor (generating actions)"""
        # extract agent features from obs
        feats, encoder_feats = self.predictor.extract_features(obs_dict, return_encoder_feats=True)
        pred_maps = self.planner.forward(None, encoder_feats=encoder_feats)
        planner_preds = SpatialPlanner.forward_prediction(
            pred_map=pred_maps,
            obs_dict=obs_dict,
            mask_drivable=kwargs["mask_drivable"],
            num_samples=kwargs["num_samples"],
            clearance=kwargs["clearance"]
        )

        plan_samples, plan_info = self.get_plan(planner_preds)

        # if evaluating multiple plan samples per obs
        b, n = plan_samples.positions.shape[:2]
        # reuse features by tiling the feature tensors to the same size as plan samples
        feats_tiled = TensorUtils.repeat_by_expand_at(feats, repeats=n, dim=0)
        # flatten the sample dimension to the batch dimension
        plan_tiled = TensorUtils.join_dimensions(plan_samples.to_dict(), begin_axis=0, end_axis=2)
        plan_tiled = Plan.from_dict(plan_tiled)

        obs_tiled = TensorUtils.repeat_by_expand_at(obs_dict, repeats=n, dim=0)
        preds = self.predictor.forward_prediction(feats_tiled, obs_tiled, plan=plan_tiled)
        preds = TensorUtils.reshape_dimensions(preds, begin_axis=0, end_axis=1, target_dims=(b, n))  # [B, N, A, ...]

        # goal-conditioning only affects ego, so use any agent prediction in the sample
        agent_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0, 1:])

        agent_pred_trajs = Trajectory(
            positions=agent_preds["predictions"]["positions"],
            yaws=agent_preds["predictions"]["yaws"]
        ).trajectories

        ego_preds = TensorUtils.map_tensor(preds, lambda x: x[:, :, 0])
        action_samples = Action(
            positions=ego_preds["predictions"]["positions"],
            yaws=ego_preds["predictions"]["yaws"]
        )
        action_sample_trajs = action_samples.trajectories

        agent_extents = obs_dict["all_other_agents_history_extents"][..., :2].max(
            dim=-2)[0]
        drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"]).float()
        dis_map = calc_distance_map(drivable_map)

        action_idx = ego_sample_planning(
            ego_trajectories=action_sample_trajs,
            agent_trajectories=agent_pred_trajs,
            ego_extents=obs_dict["extent"][:, :2],
            agent_extents=agent_extents,
            raw_types=obs_dict["all_other_agents_types"],
            raster_from_agent=obs_dict["raster_from_agent"],
            dis_map=dis_map,
            log_likelihood=plan_info["log_likelihood"],
            weights=kwargs["cost_weights"],
        )

        action_trajs_best = torch.gather(
            action_sample_trajs,
            dim=1,
            index=action_idx[:, None, None, None].expand(-1, 1, *action_sample_trajs.shape[2:])
        ).squeeze(1)

        ego_actions = Action(
            positions=action_trajs_best[..., :2], yaws=action_trajs_best[..., 2:])

        action_info = dict(
            plan_samples=plan_info["plan_samples"].to_dict(),
            plan_info=dict(location_map=plan_info["location_map"]),
            action_samples=action_samples.to_dict()
        )
        return ego_actions, action_info


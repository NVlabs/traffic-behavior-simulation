from collections import OrderedDict
from random import sample
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
from tbsim import dynamics
import tbsim.utils.torch_utils as TorchUtils
import tbsim.utils.geometry_utils as GeoUtils

from tbsim.models.rasterized_models import (
    RasterizedPlanningModel,
    RasterizedVAEModel,
    RasterizedGCModel,
    RasterizedGANModel,
    RasterizedDiscreteVAEModel,
    RasterizedECModel,
    RasterizedTreeVAEModel,
    RasterizedSceneTreeModel,
    UnetSceneTreeModel,
)
from tbsim.models.base_models import (
    MLPTrajectoryDecoder,
    RasterizedMapUNet,
)

from tbsim.models.ScePT import ScenePolicyTrajectron
from tbsim.models.agentformer import AgentFormer
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.metrics as Metrics
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.loss_utils as LossUtils
from tbsim.policies.common import Plan, Action
import tbsim.algos.algo_utils as AlgoUtils
from tbsim.utils.geometry_utils import transform_points_tensor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import wandb


class BehaviorCloning(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(BehaviorCloning, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedPlanningModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)


        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
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
        preds = self(obs_dict)
        plan = Plan(
            positions=preds["positions"],
            yaws=preds["yaws"],
            availabilities=torch.ones(preds["positions"].shape[:-1]).to(
                preds["positions"].device
            ),  # [B, T]
        )
        return plan, {}

    def get_action(self, obs_dict, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


class BehaviorCloningGC(BehaviorCloning):
    def __init__(self, algo_config, modality_shapes):
        """
        Creates networks and places them into @self.nets.
        """
        pl.LightningModule.__init__(self)
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        traj_decoder = MLPTrajectoryDecoder(
            feature_dim=algo_config.map_feature_dim + algo_config.goal_feature_dim,
            state_dim=3,
            num_steps=algo_config.future_num_frames,
            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,
            step_time=algo_config.step_time,
            network_kwargs=algo_config.decoder,
        )

        self.nets["policy"] = RasterizedGCModel(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            trajectory_decoder=traj_decoder,
            map_feature_dim=algo_config.map_feature_dim,
            weights_scaling=[1.0, 1.0, 1.0],
            goal_feature_dim=algo_config.goal_feature_dim,
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    def get_action(self, obs_dict, **kwargs):
        obs_dict = dict(obs_dict)
        if "plan" in kwargs:
            plan = kwargs["plan"]
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}


class SpatialPlanner(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(SpatialPlanner, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=4,  # (pixel, x_residual, y_residual, yaw)
            use_spatial_softmax=algo_config.spatial_softmax.enabled,
            spatial_softmax_kwargs=algo_config.spatial_softmax.kwargs,
        )

    @property
    def checkpoint_monitor_keys(self):
        keys = {"posErr": "val/metrics_goal_pos_err"}
        if self.algo_config.loss_weights.pixel_bce_loss > 0:
            keys["valBCELoss"] = "val/losses_pixel_bce_loss"
        if self.algo_config.loss_weights.pixel_ce_loss > 0:
            keys["valCELoss"] = "val/losses_pixel_ce_loss"
        return keys

    def forward(self, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        pred_map = self.nets["policy"](obs_dict["image"])
        return self.forward_prediction(
            pred_map,
            obs_dict,
            mask_drivable=mask_drivable,
            num_samples=num_samples,
            clearance=clearance
        )

    @staticmethod
    def forward_prediction(pred_map, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        assert pred_map.shape[1] == 4  # [location_logits, residual_x, residual_y, yaw]

        pred_map[:, 1:3] = torch.sigmoid(pred_map[:, 1:3])
        location_map = pred_map[:, 0]

        # get normalized probability map
        location_prob_map = torch.softmax(location_map.flatten(1), dim=1).reshape(location_map.shape)

        if mask_drivable:
            # At test time: optionally mask out undrivable regions
            if "drivable_map" not in obs_dict:
                drivable_map = batch_utils().get_drivable_region_map(obs_dict["image"])
            else:
                drivable_map = obs_dict["drivable_map"]
            for i, m in enumerate(drivable_map):
                if m.sum() == 0:  # if nowhere is drivable, set it to all True's to avoid decoding problems
                    drivable_map[i] = True

            location_prob_map = location_prob_map * drivable_map.float()
        
        prob_sum = location_prob_map.sum([1,2])
        zero_index = torch.where(prob_sum==0)[0]
        if zero_index.nelement()>0:
            location_prob_map[zero_index] = torch.ones_like(location_prob_map[zero_index])
            location_prob_map[zero_index] = location_prob_map[zero_index]/location_prob_map[zero_index].sum([1,2])

        # decode map as predictions
        pixel_pred, res_pred, yaw_pred, pred_prob = AlgoUtils.decode_spatial_prediction(
            prob_map=location_prob_map,
            residual_yaw_map=pred_map[:, 1:],
            num_samples=num_samples,
            clearance = clearance,
        )

        # transform prediction to agent coordinate
        pos_pred = transform_points_tensor(
            (pixel_pred + res_pred),
            obs_dict["agent_from_raster"].float()
        )

        return dict(
            predictions=dict(
                positions=pos_pred,
                yaws=yaw_pred
            ),
            log_likelihood=torch.log(pred_prob),
            spatial_prediction=pred_map,
            location_map=location_map,
            location_prob_map=location_prob_map
        )

    @staticmethod
    def compute_metrics(pred_batch, data_batch):
        metrics = dict()
        goal_sup = data_batch["goal"]
        goal_pred = TensorUtils.squeeze(pred_batch["predictions"], dim=1)

        pos_norm_err = torch.norm(
            goal_pred["positions"] - goal_sup["goal_position"], dim=-1
        )
        metrics["goal_pos_err"] = torch.mean(pos_norm_err)

        metrics["goal_yaw_err"] = torch.mean(
            torch.abs(goal_pred["yaws"] - goal_sup["goal_yaw"])
        )

        pixel_pred = torch.argmax(
            torch.flatten(pred_batch["location_map"], start_dim=1), dim=1
        )  # [B]
        metrics["goal_selection_err"] = torch.mean(
            (goal_sup["goal_position_pixel_flat"].long() != pixel_pred).float()
        )
        metrics["goal_cls_err"] = torch.mean((torch.exp(pred_batch["log_likelihood"]) < 0.5).float())
        metrics = TensorUtils.to_numpy(metrics)
        for k, v in metrics.items():
            metrics[k] = float(v)
        return metrics

    @staticmethod
    def compute_losses(pred_batch, data_batch):
        losses = dict()
        pred_map = pred_batch["spatial_prediction"]
        b, c, h, w = pred_map.shape

        goal_sup = data_batch["goal"]
        # compute pixel classification loss
        location_prediction = pred_map[:, 0]
        losses["pixel_bce_loss"] = torch.binary_cross_entropy_with_logits(
            input=location_prediction,  # [B, H, W]
            target=goal_sup["goal_spatial_map"],  # [B, H, W]
        ).mean()

        losses["pixel_ce_loss"] = torch.nn.CrossEntropyLoss()(
            input=location_prediction.flatten(start_dim=1),  # [B, H * W]
            target=goal_sup["goal_position_pixel_flat"].long(),  # [B]
        )

        # compute residual and yaw loss
        gather_inds = TensorUtils.unsqueeze_expand_at(
            goal_sup["goal_position_pixel_flat"].long(), size=c, dim=1
        )[..., None]  # -> [B, C, 1]

        local_pred = torch.gather(
            input=torch.flatten(pred_map, 2),  # [B, C, H * W]
            dim=2,
            index=gather_inds  # [B, C, 1]
        ).squeeze(-1)  # -> [B, C]
        residual_pred = local_pred[:, 1:3]
        yaw_pred = local_pred[:, 3:4]
        losses["pixel_res_loss"] = torch.nn.MSELoss()(residual_pred, goal_sup["goal_position_residual"])
        losses["pixel_yaw_loss"] = torch.nn.MSELoss()(yaw_pred, goal_sup["goal_yaw"])

        return losses

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.forward(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = self.compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        with torch.no_grad():
            metrics = self.compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self(batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = TensorUtils.detach(self.compute_losses(pout, batch))
        metrics = self.compute_metrics(pout, batch)
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

    def get_plan(self, obs_dict, mask_drivable=False, sample=False, num_plan_samples=1, clearance=None, **kwargs):
        num_samples = num_plan_samples if sample else None
        preds = self.forward(obs_dict, mask_drivable=mask_drivable, num_samples=num_samples,clearance=clearance)  # [B, num_sample, ...]
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

        # take the first sample as the plan
        plan = TensorUtils.map_tensor(plan_samples.to_dict(), lambda x: x[:, 0])
        plan = Plan.from_dict(plan)

        return plan, dict(location_map=preds["location_map"], plan_samples=plan_samples, log_likelihood=preds["log_likelihood"])


class VAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(VAETrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss", "minADE": "val/metrics_ego_avg_ADE"}

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
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

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan is not None and self.algo_config.goal_conditional:
            assert isinstance(plan, Plan)
            obs_dict["target_positions"] = plan.positions
            obs_dict["target_yaws"] = plan.yaws
            obs_dict["target_availabilities"] = plan.availabilities
        else:
            assert not self.algo_config.goal_conditional

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                ).to_dict()
            )
        else:
            # otherwise, use prior mean to generate the sample
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info


class DiscreteVAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(DiscreteVAETrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedDiscreteVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss", "minADE": "val/metrics_ego_avg_ADE"}

    def forward(self, obs_dict):
        return self.nets["policy"].predict(obs_dict)["predictions"]

    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=min(self.algo_config.vae.num_eval_samples,self.algo_config.vae.latent_dim))
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=min(self.algo_config.vae.num_eval_samples,self.algo_config.vae.latent_dim))

        metrics = self._compute_metrics(pout, samples, batch)
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

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])
        z1 = TensorUtils.to_numpy(torch.argmax(pred_batch["z"],dim=-1))
        prob = np.take_along_axis(TensorUtils.to_numpy(pred_batch["q"]),z1,1)
        # compute ADE & FDE based on posterior params
        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])

        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, sample_preds[:,0], avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, sample_preds[:,0], avail
        ).mean()

        # compute ADE & FDE based on trajectory samples

        fake_prob = np.ones(sample_preds.shape[:2])/sample_preds.shape[1]

        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        
        metrics["mode_max"] = prob.max(1).mean()-1/prob.shape[1]

        return metrics

    def get_metrics(self, data_batch, traj_batch=None,horizon=None):
        
        pout = self.nets["policy"](data_batch)
        bs, M = pout["x_recons"]["trajectories"].shape[:2]
        if horizon is None:
            horizon = pout["x_recons"]["trajectories"].shape[-2]
        horizon = min([horizon,pout["x_recons"]["trajectories"].shape[-2]])
        if "logvar" in pout["x_recons"]:
            horizon = min([horizon,pout["x_recons"]["logvar"].shape[-2]])
        if traj_batch is not None:
            horizon = min([traj_batch["target_positions"].shape[-2],horizon])
            GT_traj = traj_batch["target_positions"][:,:horizon].reshape(bs,-1)
        else:
            GT_traj = data_batch["target_positions"][:,:horizon].reshape(bs,-1)
        if "logvar" in pout["x_recons"]:
            var = torch.exp(pout["x_recons"]["logvar"][:,:,:horizon,:2]).reshape(bs,M,-1).clamp(min=1e-4)
        else:
            var = None
        pred_traj = pout["x_recons"]["trajectories"][:,:,:horizon,:2].reshape(bs,M,-1)
        
        self.algo_config.eval.mode="mean"
        with torch.no_grad():
            try:
                loglikelihood = Metrics.GMM_loglikelihood(GT_traj, pred_traj, var, pout["p"],mode=self.algo_config.eval.mode)
            except:
                horizon1 = min([GT_traj.shape[-1],pred_traj.shape[-1]])
                if var is not None:
                    horizon1 = min([horizon1,var.shape[-1]])
                loglikelihood = Metrics.GMM_loglikelihood(GT_traj[...,:horizon1], pred_traj[...,:horizon1], var[...,:horizon1], pout["p"],mode=self.algo_config.eval.mode)    
        return OrderedDict(loglikelihood=loglikelihood.detach())

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                ).to_dict()
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

class BehaviorCloningEC(BehaviorCloning):
    def __init__(self, algo_config, modality_shapes):
        super(BehaviorCloningEC, self).__init__(algo_config, modality_shapes)

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedECModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]


    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            losses[lk] = l * self.algo_config.loss_weights[lk]
            total_loss += losses[lk]

        metrics = self._compute_metrics(pout, batch)

        for lk, l in losses.items():
            self.log("train/losses_" + lk, l)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)
        return {
            "loss": total_loss,
            "all_losses": losses,
            "all_metrics": metrics
        }

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        batch["goal"] = AlgoUtils.get_spatial_goal_supervision(batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        metrics = self._compute_metrics(pout, batch)
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

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = {}
        predictions = pred_batch["predictions"]
        preds = TensorUtils.to_numpy(predictions["positions"])
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        ade = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, preds, avail
        )
        fde = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, preds, avail
        )

        metrics["ego_ADE"] = np.mean(ade)
        metrics["ego_FDE"] = np.mean(fde)

        targets_all = batch_utils().batch_to_target_all_agents(data_batch)
        raw_type = torch.cat(
            (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
            dim=1,
        ).type(torch.int64)
        pred_edges = batch_utils().generate_edges(
            raw_type,
            targets_all["extents"],
            pos_pred=targets_all["target_positions"],
            yaw_pred=targets_all["target_yaws"],
        )

        coll_rates = TensorUtils.to_numpy(
            Metrics.batch_pairwise_collision_rate(pred_edges))

        EC_edges,type_mask = batch_utils().gen_EC_edges(
            pred_batch["EC_trajectories"],
            pred_batch["cond_traj"],
            data_batch["extent"][...,:2],
            data_batch["all_other_agents_future_extents"][...,:2].max(dim=2)[0],
            data_batch["all_other_agents_types"]
        )
        EC_coll_rate = TensorUtils.to_numpy(Metrics.batch_pairwise_collision_rate_masked(EC_edges,type_mask))
        sample_preds = TensorUtils.to_numpy(pred_batch["EC_trajectories"][...,:2])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        EC_ade = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()

        EC_fde = metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["EC_ADE"] = EC_ade
        metrics["EC_FDE"] = EC_fde
        for c in coll_rates:
            metrics["coll_" + c] = float(coll_rates[c])
        for c in EC_coll_rate:
            metrics["EC_coll_" + c] = float(EC_coll_rate[c])

        return metrics

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan=None, **kwargs):
        preds = self(obs_dict)
        action = Action(
            positions=preds["positions"],
            yaws=preds["yaws"]
        )
        return action, {}
    def get_EC_pred(self,obs,cond_traj,goal_state=None):
        return self.nets["policy"].EC_predict(obs,cond_traj,goal_state)


class GANTrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(GANTrafficModel, self).__init__()

        self.algo_config = algo_config
        self.nets = RasterizedGANModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"egoADE": "val/metrics_ego_ADE"}

    def forward(self, obs_dict):
        return self.nets.forward(obs_dict)["predictions"]

    def _compute_metrics(self, pred_batch, data_batch, sample_batch = None):
        metrics = {}

        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])

        # compute ADE & FDE based on posterior params
        recon_preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, recon_preds, avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, recon_preds, avail
        ).mean()

        # print(metrics["ego_ADE"])

        # compute ADE & FDE based on trajectory samples
        if sample_batch is None:
            return metrics

        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        conf = np.ones(sample_preds.shape[0:2]) / float(sample_preds.shape[1])
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, conf, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, conf, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, conf, avail, "max").mean()

        return metrics

    def training_step(self, batch, batch_idx, optimizer_idx):
        # pout = self.nets(batch)
        # losses = self.nets.compute_losses(pout, batch)
        batch = batch_utils().parse_batch(batch)

        if optimizer_idx == 0:
            pout = self.nets.forward_generator(batch)
            losses = self.nets.compute_losses_generator(pout, batch)
            total_loss = 0.0
            for lk, l in losses.items():
                loss = l * self.algo_config.loss_weights[lk]
                self.log("train/losses_" + lk, loss)
                total_loss += loss
            metrics = self._compute_metrics(pout, batch)
            for mk, m in metrics.items():
                self.log("train/metrics_" + mk, m)
            # print("gen", gen_loss.item())
            return total_loss
        if optimizer_idx == 1:
            pout = self.nets.forward_discriminator(batch)
            losses = self.nets.compute_losses_discriminator(pout, batch)
            total_loss = losses["gan_disc_loss"] * self.algo_config.loss_weights["gan_disc_loss"]
            # print("disc", total_loss.item())
            self.log("train/losses_disc_loss", total_loss)
            return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = TensorUtils.detach(self.nets.forward_generator(batch))
        losses = self.nets.compute_losses_generator(pout, batch)
        with torch.no_grad():
            samples = self.nets.sample(batch, n=self.algo_config.gan.num_eval_samples)
        metrics = self._compute_metrics(pout, batch, samples)
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
        gen_optim = optim.Adam(
            params=self.nets.generator_mods.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

        optim_params = self.algo_config.optim_params["disc"]
        disc_optim = optim.Adam(
            params=self.nets.discriminator_mods.parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["regularization"]["L2"],
        )

        return [gen_optim, disc_optim], []

    def get_action(self, obs_dict, num_action_samples=1, **kwargs):
        obs_dict = dict(obs_dict)

        preds = self.nets.sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
        action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
        info = dict(
            action_samples=Action(
                positions=preds["positions"],
                yaws=preds["yaws"]
            ).to_dict()
        )

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info


class SimpleTreeModel():
    def __init__(self,algo_config):
        self.dyn=dynamics.Unicycle()
        self.algo_config = algo_config
        self.stage=algo_config.stage
        self.num_frames_per_stage = algo_config.num_frames_per_stage
        self.algo_config.ego_conditioning = False
        self.algo_config.scene_centric = False
        self.bu = batch_utils(rasterize_mode="none",parse=True)
        self.algo_config.vae.num_latent_sample = 2
        self.algo_config.M = 2
    def predict(self,obs):
        
        curr_states = self.bu.get_current_states_all_agents(obs, self.algo_config.step_time,self.dyn.type())
        bs,Na = curr_states.shape[:2]
        device = curr_states.device
        M = 2
        inputs = torch.zeros([bs,M,Na,2]).to(device)
        # inputs[:,1,:,1]=-0.02*curr_states[...,2]
        # inputs[:,2,:,1]=0.02*curr_states[...,2]
        inputs[:,1,:,0]=-3
        inputs = inputs.unsqueeze(-2).repeat_interleave(self.num_frames_per_stage,-2)
        pred_by_stage = list()
        for stage in range(self.stage):
            inputs_i = inputs.repeat_interleave(M**stage,0)
            if stage==0:
                curr_state_i = curr_states
            else:
                curr_state_i = pred_by_stage[stage-1][...,-1,:]
            pred_i,_,_ = self.dyn.forward_dynamics(curr_state_i.repeat_interleave(M,0),TensorUtils.join_dimensions(inputs_i,0,2),self.algo_config.step_time)
            pred_by_stage.append(pred_i)
        
        for stage in range(self.stage):
            pred_by_stage[stage]=pred_by_stage[stage].reshape([bs,-1,Na,self.num_frames_per_stage,4]).repeat_interleave(M**(self.stage-1-stage),1)
        pred_traj = torch.cat(pred_by_stage,-2).unsqueeze(1)[...,[0,1,3]]
        prob = torch.ones([bs,1,M**self.stage]).to(device)
        prob = prob/prob.sum(-1,keepdim=True)
        return dict(trajectories = pred_traj,p=prob)

        



class TreeVAETrafficModel(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(TreeVAETrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.nets["policy"] = RasterizedTreeVAEModel(
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            weights_scaling=[1.0, 1.0, 1.0],
        )
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def get_EC_pred(self,obs_dict,cond_traj,goal=None):
        if goal is None:
            return self.nets["policy"](obs_dict,cond_traj=cond_traj)
        else:
            return self.nets["policy"](obs_dict,cond_traj=cond_traj,goal=goal)


    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = batch_utils().parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.vae.num_eval_samples)
        metrics = self._compute_metrics(pout, samples, batch)
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

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}
        total_horizon = self.nets["policy"].stage*self.nets["policy"].num_frames_per_stage
        gt = TensorUtils.to_numpy(data_batch["target_positions"])
        gt = gt[...,:total_horizon,:]
        avail = TensorUtils.to_numpy(data_batch["target_availabilities"])
        avail = avail[...,:total_horizon]
        
        # compute ADE & FDE based on posterior params

        sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"])
        prob = TensorUtils.to_numpy(sample_batch["p"])
        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, gt, sample_preds[:,0], avail
        ).mean()
        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, gt, sample_preds[:,0], avail
        ).mean()

        # compute ADE & FDE based on trajectory samples
        
        fake_prob = prob/prob.sum(-1,keepdims=True)
        
        metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()
        metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt, sample_preds, fake_prob, avail, "oracle").mean()

        # compute diversity scores based on trajectory samples
        metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "mean").mean()
        metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt, sample_preds, fake_prob, avail, "max").mean()
        
        metrics["mode_max"] = prob.max(1).mean()-1/prob.shape[1]

        return metrics

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                )
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

class SceneCentricModel(pl.LightningModule):
    def __init__(self):
        super(SceneCentricModel, self).__init__()


    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}

    def forward(self, obs_dict):
        return self.nets["policy"](obs_dict)["predictions"]

    def predict(self,obs,**kwargs):
        with torch.no_grad():
            return TensorUtils.detach(self.nets["policy"](obs,predict=True,**kwargs))



    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = self.bu.parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        with torch.no_grad():
            samples = self.nets["policy"].sample(batch, n=self.algo_config.num_eval_samples)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        metrics = self._compute_metrics(pout, samples, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = self.bu.parse_batch(batch)
        
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))

        if self.logger is not None and batch_idx==1:
            self.log_pred_image(batch,pout)
            print("image logged")

        if "vae" in self.algo_config:
            with torch.no_grad():
                samples = self.nets["policy"].sample(batch, n=self.algo_config.num_eval_samples)
        else: 
            samples = pout
        metrics = self._compute_metrics(pout, samples, batch)

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

    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = {}
        
        total_horizon = self.nets["policy"].stage*self.nets["policy"].num_frames_per_stage
        if data_batch["target_positions"].ndim==3:
            gt = torch.cat((data_batch["target_positions"].unsqueeze(1),data_batch["all_other_agents_future_positions"]),1)
            avail = torch.cat((data_batch["target_availabilities"].unsqueeze(1),data_batch["all_other_agents_future_availability"]),1)*pred_batch["agent_avail"].unsqueeze(-1)
        elif data_batch["target_positions"].ndim==4:
            gt = data_batch["target_positions"]
            avail = data_batch["target_availabilities"]
        gt = TensorUtils.to_numpy(gt)
        gt = gt[...,:total_horizon,:]
        
        avail = TensorUtils.to_numpy(avail)
        avail = avail[...,:total_horizon]
        
        # compute ADE & FDE based on posterior params
        if sample_batch is not None:
            prob = TensorUtils.to_numpy(sample_batch["p"])    
        pred_prob = TensorUtils.to_numpy(pred_batch["p"])
        z = np.argmax(pred_prob,-1)
        preds = TensorUtils.to_numpy(pred_batch["predictions"]["positions"])
        bs,Ne,numMode,Na= preds.shape[:4]
        if pred_prob.ndim==3:
            idx = np.tile(z.reshape(bs,Ne,1,1,1,1),(1,1,1,*preds.shape[-3:]))
            pred_selected = np.take_along_axis(preds,idx,2)
        elif pred_prob.ndim==4:
            #varying with agent
            idx = np.tile(z.reshape(bs,Ne,1,Na,1,1),(1,1,1,1,*preds.shape[-2:]))
            pred_selected = np.take_along_axis(preds,idx,2)

        metrics["ego_ADE"] = Metrics.single_mode_metrics(
            Metrics.batch_average_displacement_error, 
            gt.repeat(Ne,0).reshape(-1,total_horizon,2), pred_selected.reshape(-1,total_horizon,2), avail.repeat(Ne,0).reshape(-1,total_horizon)
        ).mean()

        metrics["ego_FDE"] = Metrics.single_mode_metrics(
            Metrics.batch_final_displacement_error, 
            gt.repeat(Ne,0).reshape(-1,total_horizon,2), pred_selected.reshape(-1,total_horizon,2), avail.repeat(Ne,0).reshape(-1,total_horizon)
        ).mean()

        # compute ADE & FDE based on trajectory samples
        if prob.ndim==3:
            fake_prob = prob[:,0]/prob[:,0].sum(-1,keepdims=True)
            n = prob.shape[-1]
        else:
            fake_prob = prob[:,0,0]/prob[:,0,0].sum(-1,keepdims=True)
            n = prob.shape[-1]
        Na = preds.shape[-3]
        gt_tiled = gt.reshape(-1,total_horizon,2)
        fake_prob = np.repeat(fake_prob.reshape(-1,n),Na,0)
        
        avail_tiled = avail.reshape(-1,avail.shape[-1])
        if sample_batch is not None:
            sample_preds = TensorUtils.to_numpy(sample_batch["predictions"]["positions"][:,0])
            sample_pred_tiled = TensorUtils.join_dimensions(sample_preds.swapaxes(1,2),0,2) 

            metrics["ego_avg_ADE"] = Metrics.batch_average_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
            metrics["ego_min_ADE"] = Metrics.batch_average_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "oracle").mean()
            metrics["ego_avg_FDE"] = Metrics.batch_final_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
            metrics["ego_min_FDE"] = Metrics.batch_final_displacement_error(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "oracle").mean()

            # compute diversity scores based on trajectory samples
            metrics["ego_avg_ATD"] = Metrics.batch_average_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
            metrics["ego_max_ATD"] = Metrics.batch_average_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "max").mean()
            metrics["ego_avg_FTD"] = Metrics.batch_final_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "mean").mean()
            metrics["ego_max_FTD"] = Metrics.batch_final_diversity(gt_tiled, sample_pred_tiled, fake_prob, avail_tiled, "max").mean()
        
        metrics["mode_max"] = pred_prob.max(-1).mean()-1/pred_prob.shape[-1]
        return metrics
    
    def log_pred_image(self,batch,pred):
        keys = ["trajectories","agent_avail","cond_traj"]
        pred = {k:v for k,v in pred.items() if k in keys}
        pred = TensorUtils.to_numpy(pred)

        if batch["image"].ndim==5:
            map = batch["image"][0,0,-3:].cpu().numpy().transpose(1,2,0)
        else:
            map = batch["image"][0,-3:].cpu().numpy().transpose(1,2,0)
        traj = pred["trajectories"][...,:2]

        avail = pred["agent_avail"].astype(float)
        avail1 = avail.copy()
        avail1[avail==0]=np.nan
        raster_from_agent = batch["raster_from_agent"].cpu().numpy()
        if raster_from_agent.ndim==2:
            raster_from_agent = raster_from_agent[np.newaxis,:]
        if traj.ndim==6:
            bs,Ne,numMode,Na,T =traj.shape[:5]
            traj = traj*avail1[:,None,None,:,None,None]
        else:
            bs,numMode,Na,T =traj.shape[:4]
            traj = traj*avail1[:,None,:,None,None]
            Ne = 1
        raster_traj = GeoUtils.batch_nd_transform_points_np(traj.reshape(bs,-1,2),raster_from_agent).reshape(bs,Ne,-1,2) 

          
        cond_traj = pred["cond_traj"]

        
        
        if cond_traj is None:
            
            fig, ax = plt.subplots(figsize=(20,20))
            canvas = FigureCanvas(fig)
            ax.imshow(map)
            ax.scatter(raster_traj[0,0,...,0],raster_traj[0,0,...,1],color="c",s=1,marker="D")

        else:
            raster_cond_traj = GeoUtils.batch_nd_transform_points_np(cond_traj[...,:2].reshape(bs,-1,2),raster_from_agent).reshape(bs,Ne,-1,2)
            h = int(np.ceil(np.sqrt(Ne)))
            w = int(np.ceil(Ne/h))
            fig, ax = plt.subplots(h,w,figsize=(20*h,20*w))
            canvas = FigureCanvas(fig)
            for i in range(Ne):
                if h>1 and w>1:
                    hi=int(i/w)
                    wi=i-w*hi
                    ax[hi,wi].imshow(map)
                    ax[hi,wi].scatter(raster_traj[0,i,...,0],raster_traj[0,i,...,1],color="c",s=1,marker="D")
                    ax[hi,wi].scatter(raster_cond_traj[0,i,:,0],raster_cond_traj[0,i,:,1],color="m",s=1,marker="D")
                elif h==1 and w==1:
                    ax.imshow(map)
                    ax.scatter(raster_traj[0,i,...,0],raster_traj[0,i,...,1],color="c",s=1,marker="D")
                    ax.scatter(raster_cond_traj[0,i,:,0],raster_cond_traj[0,i,:,1],color="m",s=1,marker="D")
                else:
                    ax[i].imshow(map)
                    ax[i].scatter(raster_traj[0,i,...,0],raster_traj[0,i,...,1],color="c",s=1,marker="D")
                    ax[i].scatter(raster_cond_traj[0,i,:,0],raster_cond_traj[0,i,:,1],color="m",s=1,marker="D")
        canvas.draw()       # draw the canvas, cache the renderer

        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.logger.experiment.log({"val_pred_image":[wandb.Image(image, caption="val_pred_image")]})
        del fig,ax,image

    def get_action(self, obs_dict, sample=True, num_action_samples=1, plan_samples=None, **kwargs):
        obs_dict = dict(obs_dict)
        if plan_samples is not None and self.algo_config.goal_conditional:
            assert isinstance(plan_samples, Plan)
            obs_dict["target_positions"] = plan_samples.positions
            obs_dict["target_yaws"] = plan_samples.yaws
            obs_dict["target_availabilities"] = plan_samples.availabilities

        if sample:
            preds = self.nets["policy"].sample(obs_dict, n=num_action_samples)["predictions"]  # [B, N, T, 3]
            action_preds = TensorUtils.map_tensor(preds, lambda x: x[:, 0])  # use the first sample as the action
            info = dict(
                action_samples=Action(
                    positions=preds["positions"],
                    yaws=preds["yaws"]
                )
            )
        else:
            # otherwise, sample action from posterior
            action_preds = self.nets["policy"].predict(obs_dict)["predictions"]
            info = dict()

        action = Action(
            positions=action_preds["positions"],
            yaws=action_preds["yaws"]
        )
        return action, info

class SceneTreeTrafficModel(SceneCentricModel):
    def __init__(self, algo_config, modality_shapes):
        super(SceneTreeTrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        if algo_config.module_name=="CNN":
            self.nets["policy"] = RasterizedSceneTreeModel(
                algo_config=algo_config,
                modality_shapes=modality_shapes,
                weights_scaling=[1.0, 1.0, 1.0],
            )
            rasterize_mode=algo_config.rasterize_mode if "rasterized_mode" in algo_config else "point"
            self.bu = batch_utils(rasterize_mode=rasterize_mode,parse=True)
        elif algo_config.module_name=="Unet":
            rasterize_mode=algo_config.rasterize_mode if "rasterized_mode" in algo_config else "square"
            self.nets["policy"] = UnetSceneTreeModel(
                algo_config=algo_config,
                modality_shapes=modality_shapes,
                weights_scaling=[1.0, 1.0, 1.0],
                rasterize_mode=rasterize_mode,
            )
            
            self.bu = batch_utils(rasterize_mode=rasterize_mode,parse=True)
        self.raseterize_mode = rasterize_mode
class ScePTTrafficModel(SceneCentricModel):
    def __init__(self, algo_config, modality_shapes):
        super(ScePTTrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        
        self.bu =batch_utils(parse=True,rasterize_mode="none")



        self.nets["policy"] = ScenePolicyTrajectron(
                algo_config=algo_config,
                modality_shapes=modality_shapes,
                weights_scaling=[1.0, 1.0, 1.0],
            )

class AgentFormerTrafficModel(SceneCentricModel):
    def __init__(self, algo_config, modality_shapes):
        super(AgentFormerTrafficModel, self).__init__()
        # assert modality_shapes["image"][0] == 15

        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        
        self.bu =batch_utils(parse=True,rasterize_mode="none")

        #TODO: make it less hacky
        modality_shapes["image"] = (7,224,224)


        self.nets["policy"] = AgentFormer(
                algo_config=algo_config,
                modality_shapes=modality_shapes,
            )
        
    def _compute_metrics(self, pred_batch, sample_batch, data_batch):
        metrics = super(AgentFormerTrafficModel, self)._compute_metrics(pred_batch,sample_batch, data_batch)
        if "var" in pred_batch:
            metrics["average_var"] = pred_batch["var"].mean().item()
        return metrics
    def training_step(self, batch, batch_idx):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            batch_idx (int): training step number - required by some Algos that need
                to perform staged training and early stopping

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        batch = self.bu.parse_batch(batch)
        pout = self.nets["policy"](batch)
        losses = self.nets["policy"].compute_losses(pout, batch)
        # take samples to measure trajectory diversity
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss
        metrics = self._compute_metrics(pout, pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = self.bu.parse_batch(batch)
        
        pout = self.nets["policy"](batch)
        losses = TensorUtils.detach(self.nets["policy"].compute_losses(pout, batch))
        if self.logger is not None and batch_idx==1:
            self.log_pred_image(batch,pout)
            print("image logged")
        
        metrics = self._compute_metrics(pout, pout, batch)
        return {"losses": losses, "metrics": metrics}
    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_prediction_loss"}



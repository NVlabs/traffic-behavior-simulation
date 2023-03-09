from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tbsim.models.learned_metrics import PermuteEBM
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.models.base_models import RasterizedMapUNet, SplitMLP
from tbsim.models.Transformer import SimpleTransformer
import tbsim.algos.algo_utils as AlgoUtils


class EBMMetric(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, do_log=True):
        """
        Creates networks and places them into @self.nets.
        """
        super(EBMMetric, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self._do_log = do_log
        assert modality_shapes["image"][0] == 15

        self.nets["ebm"] = PermuteEBM(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            map_feature_dim=algo_config.map_feature_dim,
            traj_feature_dim=algo_config.traj_feature_dim,
            embedding_dim=algo_config.embedding_dim,
            embed_layer_dims=algo_config.embed_layer_dims,
        )

    @property
    def checkpoint_monitor_keys(self):
        return {"valLoss": "val/losses_infoNCE_loss"}

    def forward(self, obs_dict):
        return self.nets["ebm"](obs_dict)

    def _compute_metrics(self, pred_batch, data_batch):
        scores = pred_batch["scores"]
        pred_inds = torch.argmax(scores, dim=1)
        gt_inds = torch.arange(scores.shape[0]).to(scores.device)
        cls_acc = torch.mean((pred_inds == gt_inds).float()).item()

        return dict(cls_acc=cls_acc)

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
        pout = self.nets["ebm"](batch)
        losses = self.nets["ebm"].compute_losses(pout, batch)
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
        pout = self.nets["ebm"](batch)
        losses = TensorUtils.detach(self.nets["ebm"].compute_losses(pout, batch))
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

    def get_metrics(self, obs_dict):
        preds = self.forward(obs_dict)
        return dict(
            scores=preds["scores"].detach()
        )


class OccupancyMetric(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super(OccupancyMetric, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        self.agent_future_cond = algo_config.agent_future_cond.enabled
        if algo_config.agent_future_cond.enabled:
            self.agent_future_every_n_frame = algo_config.agent_future_cond.every_n_frame
            self.future_num_frames = int(np.floor(algo_config.future_num_frames/self.agent_future_every_n_frame))
            C,H,W = modality_shapes["image"]
            modality_shapes["image"] = (C+self.future_num_frames,H,W)

        self.nets["policy"] = RasterizedMapUNet(
            model_arch=algo_config.model_architecture,
            input_image_shape=modality_shapes["image"],  # [C, H, W]
            output_channel=algo_config.future_num_frames
        )

          

    @property
    def checkpoint_monitor_keys(self):
        keys = {"posErr": "val/metrics_pos_selection_err"}
        if self.algo_config.loss_weights.pixel_bce_loss > 0:
            keys["valBCELoss"] = "val/losses_pixel_bce_loss"
        if self.algo_config.loss_weights.pixel_ce_loss > 0:
            keys["valCELoss"] = "val/losses_pixel_ce_loss"
        return keys

    def rasterize_agent_future(self,obs_dict):

        b, t_h, h, w = obs_dict["image"].shape  # [B, C, H, W]
        t_f = self.future_num_frames

        # create spatial supervisions
        agent_positions = obs_dict["all_other_agents_future_positions"][:,:,::self.agent_future_every_n_frame]

        pos_raster = transform_points_tensor(
            agent_positions.reshape(b,-1,2),
            obs_dict["raster_from_agent"].float()
        ).reshape(b,-1,t_f,2).long()  # [B, T, 2]
        # make sure all pixels are within the raster image
        pos_raster[..., 0] = pos_raster[..., 0].clip(0, w - 1e-5)
        pos_raster[..., 1] = pos_raster[..., 1].clip(0, h - 1e-5)

        # compute flattened pixel location
        hist_image = torch.zeros([b,t_f,h*w],dtype=obs_dict["image"].dtype,device=obs_dict["image"].device)
        raster_hist_pos_flat = pos_raster[..., 1] * w + pos_raster[..., 0]  # [B, T, A]
        raster_hist_pos_flat = (raster_hist_pos_flat * obs_dict["all_other_agents_future_availability"][:,:,::self.agent_future_every_n_frame]).long()

        hist_image.scatter_(dim=2, index=raster_hist_pos_flat.transpose(1,2), src=torch.ones_like(hist_image))  # mark other agents with -1

        hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
        hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

        return hist_image.reshape(b, t_f, h, w)
        

    def forward(self, obs_dict, mask_drivable=False, num_samples=None, clearance=None):
        if self.agent_future_cond:
            hist_image = self.rasterize_agent_future(obs_dict)
            image = torch.cat([obs_dict["image"],hist_image],1)
        else:
            image = obs_dict["image"]
        
        pred_map = self.nets["policy"](image)

        return {
            "occupancy_map": pred_map
        }

    def compute_likelihood(self, occupancy_map, traj_pos, raster_from_agent):
        b, t, h, w = occupancy_map.shape  # [B, C, H, W]
        
        # create spatial supervisions
        pos_raster = transform_points_tensor(
            traj_pos,
            raster_from_agent.float()
        )  # [B, T, 2]
        # make sure all pixels are within the raster image
        pos_raster[..., 0] = pos_raster[..., 0].clip(0, w - 1e-5)
        pos_raster[..., 1] = pos_raster[..., 1].clip(0, h - 1e-5)

        pos_pixel = torch.floor(pos_raster).float()  # round down pixels

        # compute flattened pixel location
        pos_pixel_flat = pos_pixel[..., 1] * w + pos_pixel[..., 0]  # [B, T]
        occupancy_map_flat = occupancy_map.reshape(b, t, -1)

        joint_li_map = torch.softmax(occupancy_map_flat, dim=-1)  # [B, T, H * W]
        joint_li = torch.gather(joint_li_map, dim=2, index=pos_pixel_flat[:, :, None].long()).squeeze(-1)
        indep_li_map = torch.sigmoid(occupancy_map_flat)
        indep_li = torch.gather(indep_li_map, dim=2, index=pos_pixel_flat[:, :, None].long()).squeeze(-1)
        return {
            "indep_likelihood": indep_li,
            "joint_likelihood": joint_li
        }

    def _compute_losses(self, pred_batch, data_batch):
        losses = dict()
        pred_map = pred_batch["occupancy_map"]
        b, t, h, w = pred_map.shape

        spatial_sup = data_batch["spatial_sup"]
        mask = data_batch["target_availabilities"]  # [B, T]
        # compute pixel classification loss
        bce_loss = torch.binary_cross_entropy_with_logits(
            input=pred_map,  # [B, T, H, W]
            target=spatial_sup["traj_spatial_map"],  # [B, T, H, W]
        ) * mask[..., None, None]
        losses["pixel_bce_loss"] = bce_loss.mean()

        ce_loss = torch.nn.CrossEntropyLoss(reduction="none")(
            input=pred_map.reshape(b * t, h * w),
            target=spatial_sup["traj_position_pixel_flat"].long().reshape(b * t),
        ) * mask.reshape(b * t)

        losses["pixel_ce_loss"] = ce_loss.mean()

        return losses

    def _compute_metrics(self, pred_batch, data_batch):
        metrics = dict()
        spatial_sup = data_batch["spatial_sup"]

        pixel_pred = torch.argmax(
            torch.flatten(pred_batch["occupancy_map"], start_dim=2), dim=2
        )  # [B, T]
        metrics["pos_selection_err"] = torch.mean(
            (spatial_sup["traj_position_pixel_flat"].long() != pixel_pred).float()
        )

        likelihood = self.compute_likelihood(
            pred_batch["occupancy_map"],
            data_batch["target_positions"],
            data_batch["raster_from_agent"]
        )

        metrics["joint_likelihood"] = likelihood["joint_likelihood"].mean()
        metrics["indep_likelihood"] = likelihood["indep_likelihood"].mean()

        metrics = TensorUtils.to_numpy(metrics)
        for k, v in metrics.items():
            metrics[k] = float(v)
        return metrics

    def training_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self.forward(batch)
        batch["spatial_sup"] = AlgoUtils.get_spatial_trajectory_supervision(batch)
        losses = self._compute_losses(pout, batch)
        total_loss = 0.0
        for lk, l in losses.items():
            loss = l * self.algo_config.loss_weights[lk]
            self.log("train/losses_" + lk, loss)
            total_loss += loss

        with torch.no_grad():
            metrics = self._compute_metrics(pout, batch)
        for mk, m in metrics.items():
            self.log("train/metrics_" + mk, m)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch = batch_utils().parse_batch(batch)
        pout = self(batch)
        batch["spatial_sup"] = AlgoUtils.get_spatial_trajectory_supervision(batch)
        losses = TensorUtils.detach(self._compute_losses(pout, batch))
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

    def get_metrics(self, obs_dict,horizon=None):
        occup_map = self.forward(obs_dict)["occupancy_map"]
        b, t, h, w = occup_map.shape  # [B, C, H, W]
        if horizon is None:
            horizon = t
        else:
            assert horizon<=t
        li = self.compute_likelihood(occup_map, obs_dict["target_positions"], obs_dict["raster_from_agent"])
        li["joint_likelihood"] = li["joint_likelihood"][:,:horizon].mean(dim=-1).detach()
        li["indep_likelihood"] = li["indep_likelihood"][:,:horizon].mean(dim=-1).detach()

        return li
from typing import Dict

import torch
import torch.nn as nn

import tbsim.models.base_models as base_models
import tbsim.utils.tensor_utils as TensorUtils


class PermuteEBM(nn.Module):
    """Raster-based model for planning.
    """

    def __init__(
            self,
            model_arch: str,
            input_image_shape,
            map_feature_dim: int,
            traj_feature_dim: int,
            embedding_dim: int,
            embed_layer_dims: tuple
    ) -> None:

        super().__init__()
        self.map_encoder = base_models.RasterizedMapEncoder(
            model_arch=model_arch,
            input_image_shape=input_image_shape,
            feature_dim=map_feature_dim,
            use_spatial_softmax=False,
            output_activation=nn.ReLU
        )
        self.traj_encoder = base_models.RNNTrajectoryEncoder(
            trajectory_dim=3,
            rnn_hidden_size=100,
            feature_dim=traj_feature_dim
        )
        self.embed_net = base_models.MLP(
            input_dim=traj_feature_dim + map_feature_dim,
            output_dim=embedding_dim,
            output_activation=nn.ReLU,
            layer_dims=embed_layer_dims
        )
        self.score_net = nn.Linear(embedding_dim, 1)

    def forward(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image_batch = data_batch["image"]
        trajs = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)
        bs = image_batch.shape[0]

        map_feat = self.map_encoder(image_batch)  # [B, D_m]
        traj_feat = self.traj_encoder(trajs)  # [B, D_t]

        # construct contrastive samples
        map_feat_rep = TensorUtils.unsqueeze_expand_at(map_feat, size=bs, dim=1)  # [B, B, D_m]
        traj_feat_rep = TensorUtils.unsqueeze_expand_at(traj_feat, size=bs, dim=0)  # [B, B, D_t]
        cat_rep = torch.cat((map_feat_rep, traj_feat_rep), dim=-1)  # [B, B, D_m + D_t]
        ebm_rep = TensorUtils.time_distributed(cat_rep, self.embed_net)  # [B, B, D]

        # calculate embeddings and scores for InfoNCE loss
        scores = TensorUtils.time_distributed(ebm_rep, self.score_net).squeeze(-1)  # [B, B]
        out_dict = dict(features=ebm_rep, scores=scores)

        return out_dict

    def get_scores(self, data_batch):
        image_batch = data_batch["image"]
        trajs = torch.cat((data_batch["target_positions"], data_batch["target_yaws"]), dim=2)

        map_feat = self.map_encoder(image_batch)  # [B, D_m]
        traj_feat = self.traj_encoder(trajs)  # [B, D_t]
        cat_rep = torch.cat((map_feat, traj_feat), dim=-1)  # [B, D_m + D_t]
        ebm_rep = self.embed_net(cat_rep)
        scores = self.score_net(ebm_rep)
        out_dict = dict(features=ebm_rep, scores=scores)

        return out_dict

    def compute_losses(self, pred_batch, data_batch):
        scores = pred_batch["scores"]
        bs = scores.shape[0]
        labels = torch.arange(bs).to(scores.device)
        loss = nn.CrossEntropyLoss()(scores, labels)
        losses = dict(infoNCE_loss=loss)

        return losses
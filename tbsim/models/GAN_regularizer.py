import torch
from tbsim.dynamics.base import DynType
import tbsim.utils.l5_utils as L5Utils
from tbsim.models.cnn_roi_encoder import obtain_map_enc


def pred2obs(
        dyn_list,
        step_time,
        src_pos,
        src_yaw,
        src_mask,
        data_batch,
        pos_pred,
        yaw_pred,
        pred_mask,
        raw_type,
        src_lanes,
        CNNmodel,
        algo_config,
        f_steps=1,
        M=1,
):
    """generate observation for the predicted scene f_step steps into the future

    Args:
        src_pos (torch.tensor[torch.float]): xy position in src
        src_yaw (torch.tensor[torch.float]): yaw in src
        src_mask (torch.tensor[torch.bool]): mask for src
        data_batch (dict): input data dictionary
        pos_pred (torch.tensor[torch.float]): predicted xy trajectory
        yaw_pred (torch.tensor[torch.float]): predicted yaw
        pred_mask (torch.tensor[torch.bool]): mask for prediction
        raw_type (torch.tensor[torch.int]): type of agents
        src_lanes (torch.tensor[torch.float]): lane info
        f_steps (int, optional): [description]. Defaults to 1.

    Returns:
        torch.tensor[torch.float]: new src for the transformer
        torch.tensor[torch.bool]: new src mask
        torch.tensor[torch.float]: new map encoding
    """
    if pos_pred.ndim == 5:
        src_pos = src_pos.unsqueeze(1).repeat(1, M, 1, 1, 1)
        src_yaw = src_yaw.unsqueeze(1).repeat(1, M, 1, 1, 1)
        src_mask = src_mask.unsqueeze(1).repeat(1, M, 1, 1)
        pred_mask = pred_mask.unsqueeze(1).repeat(1, M, 1, 1)
        raw_type = raw_type.unsqueeze(1).repeat(1, M, 1)
    pos_new = torch.cat(
        (src_pos[..., f_steps:, :], pos_pred[..., :f_steps, :]), dim=-2
    )
    yaw_new = torch.cat(
        (src_yaw[..., f_steps:, :], yaw_pred[..., :f_steps, :]), dim=-2
    )
    src_mask_new = torch.cat(
        (src_mask[..., f_steps:], pred_mask[..., :f_steps]), dim=-1
    )
    vel_new = dyn_list[DynType.UNICYCLE].calculate_vel(
        pos_new, yaw_new, step_time, src_mask_new
    )
    src_new, _, _ = L5Utils.raw2feature(
        pos_new,
        vel_new,
        yaw_new,
        raw_type,
        src_mask_new,
        torch.zeros_like(src_lanes) if src_lanes is not None else None,
    )

    if M == 1:
        map_emb_new = obtain_map_enc(
            data_batch["image"],
            CNNmodel,
            pos_new,
            yaw_new,
            data_batch["raster_from_agent"],
            src_mask_new,
            torch.tensor(algo_config.CNN.patch_size).to(src_pos.device),
            algo_config.CNN.output_size,
            mode="last",
        )

    else:
        map_emb_new = list()
        for i in range(M):

            map_emb_new_i = obtain_map_enc(
                data_batch["image"],
                CNNmodel,
                pos_new[:, i],
                yaw_new[:, i],
                data_batch["raster_from_agent"],
                src_mask_new,
                torch.tensor(algo_config.CNN.patch_size).to(src_pos.device),
                algo_config.CNN.output_size,
                mode="last",
            )
            map_emb_new.append(map_emb_new_i)
        map_emb_new = torch.stack(map_emb_new, dim=1)
    return src_new, src_mask_new, map_emb_new


def pred2obs_static(
        dyn_list,
        step_time,
        data_batch,
        pos_pred,
        yaw_pred,
        pred_mask,
        raw_type,
        src_lanes,
        CNNmodel,
        algo_config,
        M=1,
):
    """generate observation for every step of the predictions

    Args:
        data_batch (dict): input data dictionary
        pos_pred (torch.tensor[torch.float]): predicted xy trajectory
        yaw_pred (torch.tensor[torch.float]): predicted yaw
        pred_mask (torch.tensor[torch.bool]): mask for prediction
        raw_type (torch.tensor[torch.int]): type of agents
        src_lanes (torch.tensor[torch.float]): lane info
    Returns:
        torch.tensor[torch.float]: new src for the transformer
        torch.tensor[torch.bool]: new src mask
        torch.tensor[torch.float]: new map encoding
    """
    if pos_pred.ndim == 5:
        pred_mask = pred_mask.unsqueeze(1).repeat(1, M, 1, 1)
        raw_type = raw_type.unsqueeze(1).repeat(1, M, 1)

    pred_vel = dyn_list[DynType.UNICYCLE].calculate_vel(
        pos_pred, yaw_pred, step_time, pred_mask
    )
    src_new, _, _ = L5Utils.raw2feature(
        pos_pred,
        pred_vel,
        yaw_pred,
        raw_type,
        pred_mask,
        torch.zeros_like(src_lanes) if src_lanes is not None else None,
        add_noise=True,
    )

    if M == 1:
        map_emb_new = obtain_map_enc(
            data_batch["image"],
            CNNmodel,
            pos_pred,
            yaw_pred,
            data_batch["raster_from_agent"],
            pred_mask,
            torch.tensor(algo_config.CNN.patch_size).to(pos_pred.device),
            algo_config.CNN.output_size,
            mode="all",
        )
    else:
        map_emb_new = list()
        for i in range(M):
            map_emb_new_i = obtain_map_enc(
                data_batch["image"],
                CNNmodel,
                pos_pred[:, i],
                yaw_pred[:, i],
                data_batch["raster_from_agent"],
                pred_mask,
                torch.tensor(algo_config.CNN.patch_size).to(pos_pred.device),
                algo_config.CNN.output_size,
                mode="all",
            )
            map_emb_new.append(map_emb_new_i)
        map_emb_new = torch.stack(map_emb_new, dim=1)

    return src_new, pred_mask, map_emb_new

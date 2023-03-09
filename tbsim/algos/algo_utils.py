import torch
from torch import optim as optim

import tbsim.utils.tensor_utils as TensorUtils
from tbsim import dynamics as dynamics
from tbsim.utils.batch_utils import batch_utils
from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
from tbsim.utils.l5_utils import get_last_available_index
from tbsim.utils.loss_utils import goal_reaching_loss, trajectory_loss, collision_loss


def generate_proxy_mask(orig_loc, radius,mode="L1"):
    """ mask out area near the existing samples to boost diversity

    Args:
        orig_loc (torch.tensor): original sample location, 1 for sample, 0 for background
        radius (int): radius in pixel space
        mode (str, optional): Defaults to "L1".

    Returns:
        torch.tensor[dtype=torch.bool]: mask for generating new samples
    """
    dis_map = calc_distance_map(orig_loc,max_dis = radius+1,mode=mode)
    return dis_map<=radius


def decode_spatial_prediction(prob_map, residual_yaw_map, num_samples=None, clearance=None):
    """
    Decode spatial predictions (e.g., UNet output) to a list of locations
    Args:
        prob_map (torch.Tensor): probability of each spatial location [B, H, W]
        residual_yaw_map (torch.Tensor): location residual (d_x, d_yy) and yaw of each location [B, 3, H, W]
        num_samples (int): (optional) if specified, take # of samples according to the discrete prob_map distribution.
            default is None, which is to take the max
    Returns:
        pixel_loc (torch.Tensor): coordinates of each predicted location before applying residual [B, N, 2]
        residual_pred (torch.Tensor): residual of each predicted location [B, N, 2]
        yaw_pred (torch.Tensor): yaw of each predicted location [B, N, 1]
        pixel_prob (torch.Tensor): probability of each sampled prediction [B, N]
    """
    # decode map as predictions
    b, h, w = prob_map.shape
    flat_prob_map = prob_map.flatten(start_dim=1)
    if num_samples is None:
        # if num_samples is not specified, take the maximum-probability location
        pixel_prob, pixel_loc_flat = torch.max(flat_prob_map, dim=1)
        pixel_prob = pixel_prob.unsqueeze(1)
        pixel_loc_flat = pixel_loc_flat.unsqueeze(1)
    else:
        # otherwise, use the probability map as a discrete distribution of location predictions
        dist = torch.distributions.Categorical(probs=flat_prob_map)
        if clearance is None:
            pixel_loc_flat = dist.sample((num_samples,)).permute(
                1, 0)  # [n_sample, batch] -> [batch, n_sample]
        else:
            proximity_map = torch.ones_like(prob_map,requires_grad=False)
            pixel_loc_flat = list()
            for i in range(num_samples):
                dist = torch.distributions.Categorical(probs=flat_prob_map*proximity_map.flatten(start_dim=1))
                sample_i = dist.sample()
                sample_mask = torch.zeros_like(flat_prob_map,dtype=torch.bool)
                sample_mask[torch.arange(b),sample_i]=True
                proxy_mask = generate_proxy_mask(sample_mask.reshape(-1,h,w),clearance)
                proximity_map = torch.logical_or(proximity_map,torch.logical_not(proxy_mask))
                pixel_loc_flat.append(sample_i)
            
            pixel_loc_flat = torch.stack(pixel_loc_flat,dim=1)
        pixel_prob = torch.gather(flat_prob_map, dim=1, index=pixel_loc_flat)

    local_pred = torch.gather(
        input=torch.flatten(residual_yaw_map, 2),  # [B, C, H * W]
        dim=2,
        index=TensorUtils.unsqueeze_expand_at(
            pixel_loc_flat, size=3, dim=1)  # [B, C, num_samples]
    ).permute(0, 2, 1)  # [B, C, N] -> [B, N, C]

    residual_pred = local_pred[:, :, 0:2]
    yaw_pred = local_pred[:, :, 2:3]

    pixel_loc_x = torch.remainder(pixel_loc_flat, w).float()
    pixel_loc_y = torch.floor(pixel_loc_flat.float() / float(w)).float()
    pixel_loc = torch.stack((pixel_loc_x, pixel_loc_y), dim=-1)  # [B, N, 2]

    return pixel_loc, residual_pred, yaw_pred, pixel_prob


def get_spatial_goal_supervision(data_batch):
    """Get supervision for training the spatial goal network."""
    b, _, h, w = data_batch["image"].shape  # [B, C, H, W]

    # use last available step as goal location
    goal_index = get_last_available_index(
        data_batch["target_availabilities"])[:, None, None]

    # gather by goal index
    goal_pos_agent = torch.gather(
        data_batch["target_positions"],  # [B, T, 2]
        dim=1,
        index=goal_index.expand(-1, 1,
                                data_batch["target_positions"].shape[-1])
    )  # [B, 1, 2]

    goal_yaw_agent = torch.gather(
        data_batch["target_yaws"],  # [B, T, 1]
        dim=1,
        index=goal_index.expand(-1, 1, data_batch["target_yaws"].shape[-1])
    )  # [B, 1, 1]

    # create spatial supervisions
    goal_pos_raster = transform_points_tensor(
        goal_pos_agent,
        data_batch["raster_from_agent"].float()
    ).squeeze(1)  # [B, 2]
    # make sure all pixels are within the raster image
    goal_pos_raster[:, 0] = goal_pos_raster[:, 0].clip(0, w - 1e-5)
    goal_pos_raster[:, 1] = goal_pos_raster[:, 1].clip(0, h - 1e-5)

    goal_pos_pixel = torch.floor(goal_pos_raster).float()  # round down pixels
    # compute rounding residuals (range 0-1)
    goal_pos_residual = goal_pos_raster - goal_pos_pixel
    # compute flattened pixel location
    goal_pos_pixel_flat = goal_pos_pixel[:, 1] * w + goal_pos_pixel[:, 0]
    raster_sup_flat = TensorUtils.to_one_hot(
        goal_pos_pixel_flat.long(), num_class=h * w)
    raster_sup = raster_sup_flat.reshape(b, h, w)
    return {
        "goal_position_residual": goal_pos_residual,  # [B, 2]
        "goal_spatial_map": raster_sup,  # [B, H, W]
        "goal_position_pixel": goal_pos_pixel,  # [B, 2]
        "goal_position_pixel_flat": goal_pos_pixel_flat,  # [B]
        "goal_position": goal_pos_agent.squeeze(1),  # [B, 2]
        "goal_yaw": goal_yaw_agent.squeeze(1),  # [B, 1]
        "goal_index": goal_index.reshape(b)  # [B]
    }


def get_spatial_trajectory_supervision(data_batch):
    """Get supervision for training the learned occupancy metric."""
    b, _, h, w = data_batch["image"].shape  # [B, C, H, W]
    t = data_batch["target_positions"].shape[-2]
    # create spatial supervisions
    pos_raster = transform_points_tensor(
        data_batch["target_positions"],
        data_batch["raster_from_agent"].float()
    )  # [B, T, 2]
    # make sure all pixels are within the raster image
    pos_raster[..., 0] = pos_raster[..., 0].clip(0, w - 1e-5)
    pos_raster[..., 1] = pos_raster[..., 1].clip(0, h - 1e-5)

    pos_pixel = torch.floor(pos_raster).float()  # round down pixels

    # compute flattened pixel location
    pos_pixel_flat = pos_pixel[..., 1] * w + pos_pixel[..., 0]
    raster_sup_flat = TensorUtils.to_one_hot(
        pos_pixel_flat.long(), num_class=h * w)
    raster_sup = raster_sup_flat.reshape(b, t, h, w)
    return {
        "traj_spatial_map": raster_sup,  # [B, T, H, W]
        "traj_position_pixel": pos_pixel,  # [B, T, 2]
        "traj_position_pixel_flat": pos_pixel_flat  # [B, T]
    }


def optimize_trajectories(
        init_u,
        init_x,
        target_trajs,
        target_avails,
        dynamics_model,
        step_time: float,
        data_batch=None,
        goal_loss_weight=1.0,
        traj_loss_weight=0.0,
        coll_loss_weight=0.0,
        num_optim_iterations: int = 50
):
    """An optimization-based trajectory generator"""
    curr_u = init_u.detach().clone()
    curr_u.requires_grad = True
    action_optim = optim.LBFGS(
        [curr_u], max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

    for oidx in range(num_optim_iterations):
        def closure():
            action_optim.zero_grad()

            # get trajectory with current params
            _, pos, yaw = dynamics_model.forward_dynamics(
                initial_states=init_x,
                actions=curr_u,
                step_time=step_time
            )
            curr_trajs = torch.cat((pos, yaw), dim=-1)
            # compute trajectory optimization losses
            losses = dict()
            losses["goal_loss"] = goal_reaching_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * goal_loss_weight
            losses["traj_loss"] = trajectory_loss(
                predictions=curr_trajs,
                targets=target_trajs,
                availabilities=target_avails
            ) * traj_loss_weight
            if coll_loss_weight > 0:
                assert data_batch is not None
                coll_edges = batch_utils().get_edges_from_batch(
                    data_batch,
                    ego_predictions=dict(positions=pos, yaws=yaw)
                )
                for c in coll_edges:
                    coll_edges[c] = coll_edges[c][:, :target_trajs.shape[-2]]
                vv_edges = dict(VV=coll_edges["VV"])
                if vv_edges["VV"].shape[0] > 0:
                    losses["coll_loss"] = collision_loss(
                        vv_edges) * coll_loss_weight

            total_loss = torch.hstack(list(losses.values())).sum()

            # backprop
            total_loss.backward()
            return total_loss
        action_optim.step(closure)

    final_raw_trajs, final_pos, final_yaw = dynamics_model.forward_dynamics(
        initial_states=init_x,
        actions=curr_u,
        step_time=step_time
    )
    final_trajs = torch.cat((final_pos, final_yaw), dim=-1)
    losses = dict()
    losses["goal_loss"] = goal_reaching_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )
    losses["traj_loss"] = trajectory_loss(
        predictions=final_trajs,
        targets=target_trajs,
        availabilities=target_avails
    )

    return dict(positions=final_pos, yaws=final_yaw), final_raw_trajs, curr_u, losses


def combine_ego_agent_data(batch, ego_keys, agent_keys, mask=None):
    assert len(ego_keys) == len(agent_keys)
    combined_batch = dict()
    for ego_key, agent_key in zip(ego_keys, agent_keys):
        if mask is None:
            size_dim0 = batch[agent_key].shape[0]*batch[agent_key].shape[1]
            combined_batch[ego_key] = torch.cat((batch[ego_key], batch[agent_key].reshape(
                size_dim0, *batch[agent_key].shape[2:])), dim=0)
        else:
            size_dim0 = mask.sum()
            combined_batch[ego_key] = torch.cat((batch[ego_key], batch[agent_key][mask].reshape(
                size_dim0, *batch[agent_key].shape[2:])), dim=0)
    return combined_batch


def yaw_from_pos(pos: torch.Tensor, dt, yaw_correction_speed=0.):
    """
    Compute yaws from position sequences. Optionally suppress yaws computed from low-velocity steps

    Args:
        pos (torch.Tensor): sequence of positions [..., T, 2]
        dt (float): delta timestep to compute speed
        yaw_correction_speed (float): zero out yaw change when the speed is below this threshold (noisy heading)

    Returns:
        accum_yaw (torch.Tensor): sequence of yaws [..., T-1, 1]
    """

    pos_diff = pos[..., 1:, :] - pos[..., :-1, :]
    yaw = torch.atan2(pos_diff[..., 1], pos_diff[..., 0])
    delta_yaw = torch.cat((yaw[..., [0]], yaw[..., 1:] - yaw[..., :-1]), dim=-1)
    speed = torch.norm(pos_diff, dim=-1) / dt
    delta_yaw[speed < yaw_correction_speed] = 0.
    accum_yaw = torch.cumsum(delta_yaw, dim=-1)
    return accum_yaw[..., None]

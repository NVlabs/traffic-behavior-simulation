import torch
import torch.nn.functional as F

import tbsim.dynamics as dynamics
import tbsim.utils.tensor_utils as TensorUtils
from tbsim import dynamics as dynamics
from tbsim.configs.base import ExperimentConfig


def get_agent_masks(raw_type):
    """
    PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
    ]
    """
    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)
    # veh_mask = veh_mask | ped_mask
    # ped_mask = ped_mask * 0
    return veh_mask, ped_mask


def get_dynamics_types(veh_mask, ped_mask):
    dyn_type = torch.zeros_like(veh_mask)
    dyn_type += dynamics.DynType.UNICYCLE * veh_mask
    dyn_type += dynamics.DynType.DI * ped_mask
    return dyn_type


def raw_to_features(batch_raw):
    """ map raw src into features of dim 21 """
    raw_type = batch_raw["raw_types"]
    pos = batch_raw["history_positions"]
    vel = batch_raw["history_velocities"]
    yaw = batch_raw["history_yaws"]
    mask = batch_raw["history_availabilities"]

    veh_mask, ped_mask = get_agent_masks(raw_type)

    # all vehicles, cyclists, and motorcyclists
    feature_veh = torch.cat((pos, vel, torch.cos(yaw), torch.sin(yaw)), dim=-1)

    # pedestrians and animals
    ped_feature = torch.cat(
        (pos, vel, vel * torch.sin(yaw), vel * torch.cos(yaw)), dim=-1
    )

    feature = feature_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + ped_feature * ped_mask.view([*raw_type.shape, 1, 1])

    type_embedding = F.one_hot(raw_type, 16)

    feature = torch.cat(
        (feature, type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1)),
        dim=-1,
    )
    feature = feature * mask.unsqueeze(-1)

    return feature


def raw_to_states(batch_raw):
    raw_type = batch_raw["raw_types"]
    pos = batch_raw["history_positions"]
    vel = batch_raw["history_velocities"]
    yaw = batch_raw["history_yaws"]
    avail_mask = batch_raw["history_availabilities"]

    veh_mask, ped_mask = get_agent_masks(raw_type)  # [B, (A)]

    # all vehicles, cyclists, and motorcyclists
    state_veh = torch.cat((pos, vel, yaw), dim=-1)  # [B, (A), T, S]
    # pedestrians and animals
    state_ped = torch.cat((pos, vel * torch.cos(yaw), vel * torch.sin(yaw)), dim=-1)  # [B, (A), T, S]

    state = state_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + state_ped * ped_mask.view([*raw_type.shape, 1, 1])  # [B, (A), T, S]

    # Get the current state of the agents
    num = torch.arange(0, avail_mask.shape[-1]).view(1, 1, -1).to(avail_mask.device)
    nummask = num * avail_mask
    last_idx, _ = torch.max(nummask, dim=2)
    curr_state = torch.gather(
        state, 2, last_idx[..., None, None].repeat(1, 1, 1, 4)
    )
    return state, curr_state


def batch_to_raw_ego(data_batch, step_time):
    batch_size = data_batch["history_positions"].shape[0]
    raw_type = torch.ones(batch_size).type(torch.int64).to(data_batch["history_positions"].device)  # [B, T]
    raw_type = raw_type * 3  # index for type PERCEPTION_LABEL_CAR

    src_pos = torch.flip(data_batch["history_positions"], dims=[-2])
    src_yaw = torch.flip(data_batch["history_yaws"], dims=[-2])
    src_mask = torch.flip(data_batch["history_availabilities"], dims=[-1]).bool()

    src_vel = dynamics.Unicycle.calculate_vel(pos=src_pos, yaw=src_yaw, dt=step_time, mask=src_mask)
    src_vel[:, -1] = data_batch["curr_speed"].unsqueeze(-1)

    raw = {
        "history_positions": src_pos,
        "history_velocities": src_vel,
        "history_yaws": src_yaw,
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": data_batch["extents"]
    }

    raw = TensorUtils.unsqueeze(raw, dim=1)  # Add the agent dimension
    return raw


def raw2feature(pos, vel, yaw, raw_type, mask, lanes=None, add_noise=False):
    "map raw src into features of dim 21+lane dim"

    """
    PERCEPTION_LABELS = [
    "PERCEPTION_LABEL_NOT_SET",
    "PERCEPTION_LABEL_UNKNOWN",
    "PERCEPTION_LABEL_DONTCARE",
    "PERCEPTION_LABEL_CAR",
    "PERCEPTION_LABEL_VAN",
    "PERCEPTION_LABEL_TRAM",
    "PERCEPTION_LABEL_BUS",
    "PERCEPTION_LABEL_TRUCK",
    "PERCEPTION_LABEL_EMERGENCY_VEHICLE",
    "PERCEPTION_LABEL_OTHER_VEHICLE",
    "PERCEPTION_LABEL_BICYCLE",
    "PERCEPTION_LABEL_MOTORCYCLE",
    "PERCEPTION_LABEL_CYCLIST",
    "PERCEPTION_LABEL_MOTORCYCLIST",
    "PERCEPTION_LABEL_PEDESTRIAN",
    "PERCEPTION_LABEL_ANIMAL",
    "AVRESEARCH_LABEL_DONTCARE",
    ]
    """
    dyn_type = torch.zeros_like(raw_type)
    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)
    veh_mask = veh_mask | ped_mask
    ped_mask = ped_mask * 0
    dyn_type += dynamics.DynType.UNICYCLE * veh_mask
    # all vehicles, cyclists, and motorcyclists
    if add_noise:
        pos_noise = torch.randn(pos.size(0), 1, 1, 2).to(pos.device) * 0.5
        yaw_noise = torch.randn(pos.size(0), 1, 1, 1).to(pos.device) * 0.1
        if pos.ndim == 5:
            pos_noise = pos_noise.unsqueeze(1)
            yaw_noise = yaw_noise.unsqueeze(1)
        feature_veh = torch.cat(
            (
                pos + pos_noise,
                vel,
                torch.cos(yaw + yaw_noise),
                torch.sin(yaw + yaw_noise),
            ),
            dim=-1,
        )
    else:
        feature_veh = torch.cat((pos, vel, torch.cos(yaw), torch.sin(yaw)), dim=-1)

    state_veh = torch.cat((pos, vel, yaw), dim=-1)

    # pedestrians and animals
    if add_noise:
        pos_noise = torch.randn(pos.size(0), 1, 1, 2).to(pos.device) * 0.5
        yaw_noise = torch.randn(pos.size(0), 1, 1, 1).to(pos.device) * 0.1
        if pos.ndim == 5:
            pos_noise = pos_noise.unsqueeze(1)
            yaw_noise = yaw_noise.unsqueeze(1)
        ped_feature = torch.cat(
            (
                pos + pos_noise,
                vel,
                vel * torch.sin(yaw + yaw_noise),
                vel * torch.cos(yaw + yaw_noise),
            ),
            dim=-1,
        )
    else:
        ped_feature = torch.cat(
            (pos, vel, vel * torch.sin(yaw), vel * torch.cos(yaw)), dim=-1
        )
    state_ped = torch.cat((pos, vel * torch.cos(yaw), vel * torch.sin(yaw)), dim=-1)
    state = state_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + state_ped * ped_mask.view([*raw_type.shape, 1, 1])
    dyn_type += dynamics.DynType.DI * ped_mask

    feature = feature_veh * veh_mask.view(
        [*raw_type.shape, 1, 1]
    ) + ped_feature * ped_mask.view([*raw_type.shape, 1, 1])

    type_embedding = F.one_hot(raw_type, 16)

    if pos.ndim == 4:
        if lanes is not None:
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1),
                    lanes[:, :, None, :].repeat(1, 1, feature.size(2), 1),
                ),
                dim=-1,
            )
        else:
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, feature.size(2), 1),
                ),
                dim=-1,
            )

    elif pos.ndim == 5:
        if lanes is not None:
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, 1, feature.size(-2), 1),
                    lanes[:, :, None, None, :].repeat(
                        1, feature.size(1), 1, feature.size(2), 1
                    ),
                ),
                dim=-1,
            )
        else:
            feature = torch.cat(
                (
                    feature,
                    type_embedding.unsqueeze(-2).repeat(1, 1, 1, feature.size(-2), 1),
                ),
                dim=-1,
            )
    feature = feature * mask.unsqueeze(-1)
    return feature, dyn_type, state


def batch_to_vectorized_feature(data_batch, dyn_list, step_time, algo_config):
    device = data_batch["history_positions"].device
    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)
    extents = torch.cat(
        (
            data_batch["extent"][..., :2].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    src_pos = torch.cat(
        (
            data_batch["history_positions"].unsqueeze(1),
            data_batch["all_other_agents_history_positions"],
        ),
        dim=1,
    )
    "history position and yaw need to be flipped so that they go from past to recent"
    src_pos = torch.flip(src_pos, dims=[-2])
    src_yaw = torch.cat(
        (
            data_batch["history_yaws"].unsqueeze(1),
            data_batch["all_other_agents_history_yaws"],
        ),
        dim=1,
    )
    src_yaw = torch.flip(src_yaw, dims=[-2])
    src_world_yaw = src_yaw + (
        data_batch["yaw"]
        .view(-1, 1, 1, 1)
        .repeat(1, src_yaw.size(1), src_yaw.size(2), 1)
    ).type(torch.float)
    src_mask = torch.cat(
        (
            data_batch["history_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_history_availability"],
        ),
        dim=1,
    ).bool()

    src_mask = torch.flip(src_mask, dims=[-1])
    # estimate velocity
    src_vel = dyn_list[dynamics.DynType.UNICYCLE].calculate_vel(
        src_pos, src_yaw, step_time, src_mask
    )

    src_vel[:, 0, -1] = torch.clip(
        data_batch["curr_speed"].unsqueeze(-1),
        min=algo_config.vmin,
        max=algo_config.vmax,
    )
    if algo_config.vectorize_lane:
        src_lanes = torch.cat(
            (
                data_batch["ego_lanes"].unsqueeze(1),
                data_batch["all_other_agents_lanes"],
            ),
            dim=1,
        ).type(torch.float)
        src_lanes = torch.cat((
            src_lanes[...,0:2],
            torch.cos(src_lanes[...,2:3]),
            torch.sin(src_lanes[...,2:3]),
            src_lanes[...,-1:],
        ),dim=-1)
        src_lanes = src_lanes.view(*src_lanes.shape[:2], -1)
    else:
        src_lanes = None
    src, dyn_type, src_state = raw2feature(
        src_pos, src_vel, src_yaw, raw_type, src_mask, src_lanes
    )
    tgt_mask = torch.cat(
        (
            data_batch["target_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_future_availability"],
        ),
        dim=1,
    ).bool()
    num = torch.arange(0, src_mask.shape[2]).view(1, 1, -1).to(src_mask.device)
    nummask = num * src_mask
    last_idx, _ = torch.max(nummask, dim=2)
    curr_state = torch.gather(
        src_state, 2, last_idx[..., None, None].repeat(1, 1, 1, 4)
    )

    tgt_pos = torch.cat(
        (
            data_batch["target_positions"].unsqueeze(1),
            data_batch["all_other_agents_future_positions"],
        ),
        dim=1,
    )
    tgt_yaw = torch.cat(
        (
            data_batch["target_yaws"].unsqueeze(1),
            data_batch["all_other_agents_future_yaws"],
        ),
        dim=1,
    )
    tgt_pos_yaw = torch.cat((tgt_pos, tgt_yaw), dim=-1)
    

    # curr_pos_yaw = torch.cat((curr_state[..., 0:2], curr_yaw), dim=-1)

    # tgt = tgt - curr_pos_yaw.repeat(1, 1, tgt.size(2), 1) * tgt_mask.unsqueeze(-1)


    return (
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
    )

def obtain_goal_state(tgt_pos_yaw,tgt_mask):
    num = torch.arange(0, tgt_mask.shape[2]).view(1, 1, -1).to(tgt_mask.device)
    nummask = num * tgt_mask
    last_idx, _ = torch.max(nummask, dim=2, keepdim=True)
    last_mask = nummask.ge(last_idx)
    
    goal_mask = tgt_mask*last_mask
    goal_pos_yaw = tgt_pos_yaw*goal_mask.unsqueeze(-1)
    return goal_pos_yaw[...,:2], goal_pos_yaw[...,2:], goal_mask


def batch_to_raw_all_agents(data_batch, step_time):
    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)

    src_pos = torch.cat(
        (
            data_batch["history_positions"].unsqueeze(1),
            data_batch["all_other_agents_history_positions"],
        ),
        dim=1,
    )
    # history position and yaw need to be flipped so that they go from past to recent
    src_pos = torch.flip(src_pos, dims=[-2])
    src_yaw = torch.cat(
        (
            data_batch["history_yaws"].unsqueeze(1),
            data_batch["all_other_agents_history_yaws"],
        ),
        dim=1,
    )
    src_yaw = torch.flip(src_yaw, dims=[-2])
    src_mask = torch.cat(
        (
            data_batch["history_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_history_availability"],
        ),
        dim=1,
    ).bool()

    src_mask = torch.flip(src_mask, dims=[-1])

    extents = torch.cat(
        (
            data_batch["extent"][..., :2].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    # estimate velocity
    src_vel = dynamics.Unicycle.calculate_vel(src_pos, src_yaw, step_time, src_mask)
    src_vel[:, 0, -1] = data_batch["curr_speed"].unsqueeze(-1)

    return {
        "history_positions": src_pos,
        "history_yaws": src_yaw,
        "curr_speed": src_vel[:, :, -1, 0],
        "raw_types": raw_type,
        "history_availabilities": src_mask,
        "extents": extents,
    }


def batch_to_target_all_agents(data_batch):
    pos = torch.cat(
        (
            data_batch["target_positions"].unsqueeze(1),
            data_batch["all_other_agents_future_positions"],
        ),
        dim=1,
    )
    yaw = torch.cat(
        (
            data_batch["target_yaws"].unsqueeze(1),
            data_batch["all_other_agents_future_yaws"],
        ),
        dim=1,
    )
    avails = torch.cat(
        (
            data_batch["target_availabilities"].unsqueeze(1),
            data_batch["all_other_agents_future_availability"],
        ),
        dim=1,
    )

    extents = torch.cat(
        (
            data_batch["extent"][..., :2].unsqueeze(1),
            torch.max(data_batch["all_other_agents_history_extents"], dim=-2)[0],
        ),
        dim=1,
    )

    return {
        "target_positions": pos,
        "target_yaws": yaw,
        "target_availabilities": avails,
        "extents": extents
    }


def generate_edges(
        raw_type,
        extents,
        pos_pred,
        yaw_pred,
        edge_mask = None,
):
    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)

    agent_mask = veh_mask | ped_mask
    edge_types = ["VV", "VP", "PV", "PP"]
    edges = {et: list() for et in edge_types}
    for i in range(agent_mask.shape[0]):
        agent_idx = torch.where(agent_mask[i] != 0)[0]
        edge_idx = torch.combinations(agent_idx, r=2)
        VV_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        VP_idx = torch.where(
            veh_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        PV_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & veh_mask[i, edge_idx[:, 1]]
        )[0]
        PP_idx = torch.where(
            ped_mask[i, edge_idx[:, 0]] & ped_mask[i, edge_idx[:, 1]]
        )[0]
        if pos_pred.ndim == 4:
            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, edge_idx[:, 0], :],
                    yaw_pred[i, edge_idx[:, 0], :],
                    pos_pred[i, edge_idx[:, 1], :],
                    yaw_pred[i, edge_idx[:, 1], :],
                    extents[i, edge_idx[:, 0]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                    extents[i, edge_idx[:, 1]]
                        .unsqueeze(-2)
                        .repeat(1, pos_pred.size(-2), 1),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[VV_idx])
            edges["VP"].append(edges_of_all_types[VP_idx])
            edges["PV"].append(edges_of_all_types[PV_idx])
            edges["PP"].append(edges_of_all_types[PP_idx])
        elif pos_pred.ndim == 5:

            edges_of_all_types = torch.cat(
                (
                    pos_pred[i, :, edge_idx[:, 0], :],
                    yaw_pred[i, :, edge_idx[:, 0], :],
                    pos_pred[i, :, edge_idx[:, 1], :],
                    yaw_pred[i, :, edge_idx[:, 1], :],
                    extents[i, None, edge_idx[:, 0], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                    extents[i, None, edge_idx[:, 1], None, :].repeat(
                        pos_pred.size(1), 1, pos_pred.size(-2), 1
                    ),
                ),
                dim=-1,
            )
            edges["VV"].append(edges_of_all_types[:, VV_idx])
            edges["VP"].append(edges_of_all_types[:, VP_idx])
            edges["PV"].append(edges_of_all_types[:, PV_idx])
            edges["PP"].append(edges_of_all_types[:, PP_idx])
    if pos_pred.ndim == 4:
        for et, v in edges.items():
            edges[et] = torch.cat(v, dim=0)
    elif pos_pred.ndim == 5:
        for et, v in edges.items():
            edges[et] = torch.cat(v, dim=1)
    return edges

def gen_edges_masked(raw_type,
                    extents,
                    pred):

    B,Na,T = pred.shape[:3]

    veh_mask = (raw_type >= 3) & (raw_type <= 13)
    ped_mask = (raw_type == 14) | (raw_type == 15)

    edges = torch.zeros([B,Na,Na,T,10]).to(pred.device)
    edges[...,:3] = pred.unsqueeze(2)
    edges[...,3:6] = pred.unsqueeze(1)
    edges[...,6:8] = extents.unsqueeze(2).unsqueeze(3)
    edges[...,8:] = extents.unsqueeze(1).unsqueeze(3)
    self_mask = ~torch.eye(Na,dtype=bool,device=raw_type.device).unsqueeze(0)
    VV_mask = torch.logical_and(veh_mask.unsqueeze(2),veh_mask.unsqueeze(1))
    VV_mask = torch.logical_and(self_mask,VV_mask)

    VP_mask = torch.logical_and(veh_mask.unsqueeze(2),ped_mask.unsqueeze(1))
    VP_mask = torch.logical_and(self_mask,VP_mask)

    PV_mask = torch.logical_and(ped_mask.unsqueeze(2),veh_mask.unsqueeze(1))
    PV_mask = torch.logical_and(self_mask,PV_mask)

    PP_mask = torch.logical_and(ped_mask.unsqueeze(2),ped_mask.unsqueeze(1))
    PP_mask = torch.logical_and(self_mask,PP_mask)


    type_mask = dict(
        VV = VV_mask,
        VP = VP_mask,
        PV = PV_mask,
        PP = PP_mask
    )
    return edges,type_mask


def gen_ego_edges(ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types):
    """generate edges between ego trajectory samples and agent trajectories

    Args:
        ego_trajectories (torch.Tensor): [B,N,T,3]
        agent_trajectories (torch.Tensor): [B,A,T,3] or [B,N,A,T,3]
        ego_extents (torch.Tensor): [B,2]
        agent_extents (torch.Tensor): [B,A,2]
        raw_types (torch.Tensor): [B,A]
    Returns:
        edges (torch.Tensor): [B,N,A,T,10]
        type_mask (dict)
    """
    B,N,T = ego_trajectories.shape[:3]
    A = agent_trajectories.shape[-3]

    veh_mask = (raw_types >= 3) & (raw_types <= 13)
    ped_mask = (raw_types == 14) | (raw_types == 15)

    edges = torch.zeros([B,N,A,T,10]).to(ego_trajectories.device)
    edges[...,:3] = ego_trajectories.unsqueeze(2).repeat(1,1,A,1,1)
    if agent_trajectories.ndim==4:
        edges[...,3:6] = agent_trajectories.unsqueeze(1).repeat(1,N,1,1,1)
    else:
        edges[...,3:6] = agent_trajectories
    edges[...,6:8] = ego_extents.reshape(B,1,1,1,2).repeat(1,N,A,T,1)
    edges[...,8:] = agent_extents.reshape(B,1,A,1,2).repeat(1,N,1,T,1)
    type_mask = {"VV":veh_mask,"VP":ped_mask}
    return edges,type_mask


def gen_EC_edges(ego_trajectories,agent_trajectories,ego_extents, agent_extents, raw_types,mask=None):
    """generate edges between ego trajectory samples and agent trajectories

    Args:
        ego_trajectories (torch.Tensor): [B,A,T,3]
        agent_trajectories (torch.Tensor): [B,A,T,3]
        ego_extents (torch.Tensor): [B,2]
        agent_extents (torch.Tensor): [B,A,2]
        raw_types (torch.Tensor): [B,A]
        mask (optional, torch.Tensor): [B,A]
    Returns:
        edges (torch.Tensor): [B,N,A,T,10]
        type_mask (dict)
    """

    B,A = ego_trajectories.shape[:2]
    T = ego_trajectories.shape[-2]

    veh_mask = (raw_types >= 3) & (raw_types <= 13)
    ped_mask = (raw_types == 14) | (raw_types == 15)

    
    if ego_trajectories.ndim==4:
        edges = torch.zeros([B,A,T,10]).to(ego_trajectories.device)
        edges[...,:3] = ego_trajectories
        edges[...,3:6] = agent_trajectories
        edges[...,6:8] = ego_extents.reshape(B,1,1,2).repeat(1,A,T,1)
        edges[...,8:] = agent_extents.unsqueeze(2).repeat(1,1,T,1)
    elif ego_trajectories.ndim==5:
        
        K = ego_trajectories.shape[2]
        edges = torch.zeros([B,A*K,T,10]).to(ego_trajectories.device)
        edges[...,:3] = TensorUtils.join_dimensions(ego_trajectories,1,3)
        edges[...,3:6] = agent_trajectories.repeat(1,K,1,1)
        edges[...,6:8] = ego_extents.reshape(B,1,1,2).repeat(1,A*K,T,1)
        edges[...,8:] = agent_extents.unsqueeze(2).repeat(1,K,T,1)
        veh_mask = veh_mask.tile(1,K)
        ped_mask = ped_mask.tile(1,K)
    if mask is not None:
        veh_mask = veh_mask*mask
        ped_mask = ped_mask*mask
    type_mask = {"VV":veh_mask,"VP":ped_mask}
    return edges,type_mask
    

def get_edges_from_batch(data_batch, ego_predictions=None, all_predictions=None):
    raw_type = torch.cat(
        (data_batch["type"].unsqueeze(1), data_batch["all_other_agents_types"]),
        dim=1,
    ).type(torch.int64)

    # Use predicted ego position to compute future box edges

    targets_all = batch_to_target_all_agents(data_batch)
    if ego_predictions is not None:
        targets_all["target_positions"] [:, 0, :, :] = ego_predictions["positions"]
        targets_all["target_yaws"][:, 0, :, :] = ego_predictions["yaws"]
    elif all_predictions is not None:
        targets_all["target_positions"] = all_predictions["positions"]
        targets_all["target_yaws"] = all_predictions["yaws"]
    else:
        raise ValueError("Please specify either ego prediction or all predictions")

    pred_edges = generate_edges(
        raw_type, targets_all["extents"],
        pos_pred=targets_all["target_positions"],
        yaw_pred=targets_all["target_yaws"]
    )
    return pred_edges


def get_last_available_index(avails):
    """
    Args:
        avails (torch.Tensor): target availabilities [B, (A), T]

    Returns:
        last_indices (torch.Tensor): index of the last available frame
    """
    num_frames = avails.shape[-1]
    inds = torch.arange(0, num_frames).to(avails.device)  # [T]
    inds = (avails > 0).float() * inds  # [B, (A), T] arange indices with unavailable indices set to 0
    last_inds = inds.max(dim=-1)[1]  # [B, (A)] calculate the index of the last availale frame
    return last_inds


def get_current_states(batch: dict, dyn_type: dynamics.DynType) -> torch.Tensor:
    bs = batch["curr_speed"].shape[0]
    if dyn_type == dynamics.DynType.BICYCLE:
        current_states = torch.zeros(bs, 6).to(batch["curr_speed"].device)  # [x, y, yaw, vel, dh, veh_len]
        current_states[:, 3] = batch["curr_speed"].abs()
        current_states[:, [4]] = (batch["history_yaws"][:, 0] - batch["history_yaws"][:, 1]).abs()
        current_states[:, 5] = batch["extent"][:, 0]  # [veh_len]
    else:
        current_states = torch.zeros(bs, 4).to(batch["curr_speed"].device)  # [x, y, vel, yaw]
        current_states[:, 2] = batch["curr_speed"]
    return current_states


def get_current_states_all_agents(batch: dict, step_time, dyn_type: dynamics.DynType) -> torch.Tensor:
    if batch["history_positions"].ndim==3:
        state_all = batch_to_raw_all_agents(batch, step_time)
    else:
        state_all = batch
    bs, na = state_all["curr_speed"].shape[:2]
    if dyn_type == dynamics.DynType.BICYCLE:
        current_states = torch.zeros(bs, na, 6).to(state_all["curr_speed"].device)  # [x, y, yaw, vel, dh, veh_len]
        current_states[:, :, :2] = state_all["history_positions"][:, :, 0]
        current_states[:, :, 3] = state_all["curr_speed"].abs()
        current_states[:, :, [4]] = (state_all["history_yaws"][:, :, 0] - state_all["history_yaws"][:, :, 1]).abs()
        current_states[:, :, 5] = state_all["extent"][:, :, 0]  # [veh_len]
    else:
        current_states = torch.zeros(bs, na, 4).to(state_all["curr_speed"].device)  # [x, y, vel, yaw]
        current_states[:, :, :2] = state_all["history_positions"][:, :, 0]
        current_states[:, :, 2] = state_all["curr_speed"]
        current_states[:,:,3:] = state_all["history_yaws"][:,:,0]
    return current_states


def get_drivable_region_map(rasterized_map):
    return rasterized_map[..., -3, :, :] < 1.


def get_modality_shapes(cfg: ExperimentConfig):
    assert cfg.env.rasterizer.map_type == "py_semantic"
    num_channels = (cfg.algo.history_num_frames + 1) * 2 + 3
    h, w = cfg.env.rasterizer.raster_size
    return dict(image=(num_channels, h, w))
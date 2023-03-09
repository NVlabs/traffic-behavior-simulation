from logging import raiseExceptions
from signal import raise_signal
import torch
import torch.nn.functional as F
import numpy as np

import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.configs.base import ExperimentConfig
from trajdata.data_structures.state import StateTensor,StateArray




def trajdata2posyawspeed(state, nan_to_zero=True):
    """Converts trajdata's state format to pos, yaw, and speed. Set Nans to 0s"""
    if isinstance(state,StateTensor):
        pos = state.position.as_tensor()
        yaw = state.heading
        speed = state.as_format("v_lon")[...,0].as_tensor()
    elif isinstance(state,StateArray):
        pos = state.position.as_ndarray()
        yaw = state.heading
        speed = state.as_format("v_lon")[...,0].as_ndarray()
    else:
        if state.shape[-1] == 7:  # x, y, vx, vy, ax, ay, sin(heading), cos(heading)
            # state = torch.cat((state[...,:6],torch.sin(state[...,6:7]),torch.cos(state[...,6:7])),-1)
            yaw = state[...,6:7]
        else:
            assert state.shape[-1] == 8
            yaw = torch.atan2(state[..., [-2]], state[..., [-1]])
        pos = state[..., :2]
        
        speed = torch.norm(state[..., 2:4], dim=-1)
    mask = torch.bitwise_not(torch.cat([pos,speed[...,None],yaw],-1).isnan().any(-1))
    if nan_to_zero:
        pos[torch.bitwise_not(mask)] = 0.
        yaw[torch.bitwise_not(mask)] = 0.
        speed[torch.bitwise_not(mask)] = 0.
    return pos, yaw, speed, mask

def rasterize_agents_scene(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    
    b, a, t, _ = agent_hist_pos.shape
    _, _, _, h, w = maps.shape
    maps = maps.clone()
    agent_hist_pos = TensorUtils.unsqueeze_expand_at(agent_hist_pos,a,1)
    agent_mask_tiled = TensorUtils.unsqueeze_expand_at(agent_mask,a,1)*TensorUtils.unsqueeze_expand_at(agent_mask,a,2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos.reshape(b*a,-1,2), raster_from_agent.reshape(b*a,3,3)).reshape(b,a,a,t,2)
    raster_hist_pos = raster_hist_pos * agent_mask_tiled.unsqueeze(-1)  # Set invalid positions to 0.0 Will correct below
    
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels [B, A, A, T, 2]
    raster_hist_pos = raster_hist_pos.transpose(2,3)
    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, A, T, A]
    hist_image = torch.zeros(b, a, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, A, T, H * W]
    
    ego_mask = torch.zeros_like(raster_hist_pos_flat,dtype=torch.bool)
    ego_mask[:,range(a),:,range(a)]=1
    agent_mask = torch.logical_not(ego_mask)


    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*agent_mask, src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=3, index=raster_hist_pos_flat*ego_mask, src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[..., 0] = 0  # correct the 0th index from invalid positions
    hist_image[..., -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, a, t, h, w)

    maps = torch.cat((hist_image, maps), dim=2)  # treat time as extra channels
    return maps


def rasterize_agents(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        filter=None,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    b, a, t, _ = agent_hist_pos.shape
    _, _, h, w = maps.shape
    

    agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
    raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
    raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below
    raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
    raster_hist_pos[..., 0].clip_(0, (w - 1))
    raster_hist_pos[..., 1].clip_(0, (h - 1))
    raster_hist_pos = torch.round(raster_hist_pos).long()  # round pixels

    raster_hist_pos_flat = raster_hist_pos[..., 1] * w + raster_hist_pos[..., 0]  # [B, T, A]

    hist_image = torch.zeros(b, t, h * w, dtype=maps.dtype, device=maps.device)  # [B, T, H * W]

    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, 1:], src=torch.ones_like(hist_image) * -1)  # mark other agents with -1
    hist_image.scatter_(dim=2, index=raster_hist_pos_flat[:, :, [0]], src=torch.ones_like(hist_image))  # mark ego with 1.
    hist_image[:, :, 0] = 0  # correct the 0th index from invalid positions
    hist_image[:, :, -1] = 0  # correct the maximum index caused by out of bound locations

    hist_image = hist_image.reshape(b, t, h, w)
    if filter=="0.5-1-0.5":
        kernel = torch.tensor([[0.5, 0.5, 0.5],
                    [0.5, 1., 0.5],
                    [0.5, 0.5, 0.5]]).to(hist_image.device)

        kernel = kernel.view(1, 1, 3, 3).repeat(t, t, 1, 1)
        hist_image = F.conv2d(hist_image, kernel,padding=1)
    if cat:
        maps = maps.clone()
        maps = torch.cat((hist_image, maps), dim=1)  # treat time as extra channels
        return maps
    else:
        return hist_image

def rasterize_agents_rec(
        maps: torch.Tensor,
        agent_hist_pos: torch.Tensor,
        agent_hist_yaw: torch.Tensor,
        agent_extent: torch.Tensor,
        agent_mask: torch.Tensor,
        raster_from_agent: torch.Tensor,
        map_res: torch.Tensor,
        cat=True,
        ego_neg = False,
        parallel_raster=True,
) -> torch.Tensor:
    """Paint agent histories onto an agent-centric map image"""
    with torch.no_grad():
        b, a, t, _ = agent_hist_pos.shape
        _, _, h, w = maps.shape
        
        coord_tensor = torch.cat((torch.arange(w).view(w,1,1).repeat_interleave(h,1),
                                torch.arange(h).view(1,h,1).repeat_interleave(w,0),),-1).to(maps.device)

        agent_hist_pos = agent_hist_pos.reshape(b, a * t, 2)
        raster_hist_pos = transform_points_tensor(agent_hist_pos, raster_from_agent)
        

        raster_hist_pos[~agent_mask.reshape(b, a * t)] = 0.0  # Set invalid positions to 0.0 Will correct below

        raster_hist_pos = raster_hist_pos.reshape(b, a, t, 2).permute(0, 2, 1, 3)  # [B, T, A, 2]
        
        raster_hist_pos_yx = torch.cat((raster_hist_pos[...,1:],raster_hist_pos[...,0:1]),-1)

        if parallel_raster:
        # vectorized version, uses much more memory
            coord_tensor_tiled = coord_tensor.view(1,1,1,h,w,-1).repeat(b,t,a,1,1,1)
            dyx = raster_hist_pos_yx[...,None,None,:]-coord_tensor_tiled
            cos_yaw = torch.cos(-agent_hist_yaw)
            sin_yaw = torch.sin(-agent_hist_yaw)

            rotM = torch.stack(
                [
                    torch.stack([cos_yaw, sin_yaw], dim=-1),
                    torch.stack([-sin_yaw, cos_yaw], dim=-1),
                ],dim=-2,
            )
            rotM = rotM.transpose(1,2)
            rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            extent_tiled = agent_extent_yx[:,None,:,None,None]
            
            flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)

            agent_mask_tiled = agent_mask.transpose(1,2).type(torch.int)
            if ego_neg:
                # flip the value for ego
                agent_mask_tiled[:,:,0] = -agent_mask_tiled[:,:,0]
            hist_img = flag*agent_mask_tiled.view(b,t,a,1,1)
            
            if hist_img.shape[2]>1:
            # aggregate along the agent dimension
                hist_img = hist_img[:,:,0] + hist_img[:,:,1:].max(2)[0]*(hist_img[:,:,0]==0)
            else:
                hist_img = hist_img.squeeze(2)
        else:

        # loop through all agents, slow but memory efficient
            coord_tensor_tiled = coord_tensor.view(1,1,h,w,-1).repeat(b,t,1,1,1)
            agent_extent_yx = torch.cat((agent_extent[...,1:2],agent_extent[...,0:1]),-1)
            hist_img_ego = torch.zeros([b,t,h,w]).to(maps.device)
            hist_img_nb = torch.zeros([b,t,h,w]).to(maps.device)
            for i in range(raster_hist_pos_yx.shape[-2]):
                dyx = raster_hist_pos_yx[...,i,None,None,:]-coord_tensor_tiled
                yaw_i = agent_hist_yaw[:,i]
                cos_yaw = torch.cos(-yaw_i)
                sin_yaw = torch.sin(-yaw_i)

                rotM = torch.stack(
                    [
                        torch.stack([cos_yaw, sin_yaw], dim=-1),
                        torch.stack([-sin_yaw, cos_yaw], dim=-1),
                    ],dim=-2,
                )
                
                rel_yx = torch.matmul(rotM.unsqueeze(-3).repeat(1,1,h,w,1,1),dyx.unsqueeze(-1)).squeeze(-1)
                extent_tiled = agent_extent_yx[:,None,i,None,None]
                
                flag = (torch.abs(rel_yx)<extent_tiled).all(-1).type(torch.int)
                if i==0:
                    if ego_neg:
                        hist_img_ego = -flag*agent_mask[:,0,:,None,None]
                    else:
                        hist_img_ego = flag*agent_mask[:,0,:,None,None]
                else:
                    hist_img_nb = torch.maximum(hist_img_nb,agent_mask[:,0,:,None,None]*flag)
                
            if a>1:
                hist_img = hist_img_ego + hist_img_nb*(hist_img_ego==0)
            else:
                hist_img = hist_img_ego

        if cat:
            maps = maps.clone()
            maps = torch.cat((hist_img, maps), dim=1)  # treat time as extra channels
            return maps
        else:
            return hist_img



def get_drivable_region_map(maps):
    if isinstance(maps, torch.Tensor):
        drivable = torch.amax(maps[..., -3:, :, :], dim=-3).bool()
    else:
        drivable = np.amax(maps[..., -3:, :, :], axis=-3).astype(bool)
    return drivable


def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch["agent_hist"].shape[1]
    fut_len = batch["agent_fut"].shape[1]
    b, a, neigh_len, _ = batch["neigh_hist"].shape
    empty_neighbor = a == 0
    if empty_neighbor:
        batch["neigh_hist"] = torch.ones(b, 1, hist_len, batch["neigh_hist"].shape[-1]) * torch.nan
        batch["neigh_fut"] = torch.ones(b, 1, fut_len, batch["neigh_fut"].shape[-1]) * torch.nan
        batch["neigh_types"] = torch.zeros(b, 1)
        batch["neigh_hist_extents"] = torch.zeros(b, 1, hist_len, batch["neigh_hist_extents"].shape[-1])
        batch["neigh_fut_extents"] = torch.zeros(b, 1, fut_len, batch["neigh_hist_extents"].shape[-1])
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch["neigh_hist"].shape[-1]) * torch.nan
        batch["neigh_hist"] = torch.cat((hist_pad, batch["neigh_hist"]), dim=2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch["neigh_hist_extents"].shape[-1])
        batch["neigh_hist_extents"] = torch.cat((hist_pad, batch["neigh_hist_extents"]), dim=2)

def parse_scene_centric(batch: dict, rasterize_mode:str):
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])

    curr_pos = hist_pos[:,:,-1]
    curr_yaw = hist_yaw[:,:,-1]
    assert isinstance(batch["centered_agent_state"],StateTensor) or isinstance(batch["centered_agent_state"],StateArray)


    curr_speed = hist_speed[..., -1]
    centered_state = batch["centered_agent_state"]
    centered_yaw = centered_state.heading[...,0]
    centered_pos = centered_state.position
    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.


    centered_world_from_agent = torch.inverse(batch["centered_agent_from_world_tf"])



    # map-related
    if batch["maps"] is not None:
        map_res = batch["maps_resolution"][0,0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        
        centered_raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(centered_state.device)
        b,a = curr_yaw.shape[:2]
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        
        agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
        center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
        raster_from_center = centered_raster_from_agent @ agents_from_center
        center_from_raster = center_from_agents @ centered_agent_from_raster

        raster_from_world = batch["rasters_from_world_tf"]
        world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
        raster_from_world[torch.isnan(raster_from_world)] = 0.
        world_from_raster[torch.isnan(world_from_raster)] = 0.
        
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
            maps = rasterize_agents_scene(
                batch["maps"],
                hist_pos,
                hist_yaw,
                None,
                hist_mask,
                raster_from_center,
                map_res
            )
        elif rasterize_mode=="square":
            #TODO: add the square rasterization function for scene-centric data
            raise NotImplementedError
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0


    d = dict(
        image=maps,
        map_names = batch["map_names"],
        drivable_map=drivable_map,
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=centered_pos,
        yaw=centered_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=centered_raster_from_agent,
        agent_from_raster=centered_agent_from_raster,
        raster_from_center=raster_from_center,
        center_from_raster=center_from_raster,
        agents_from_center = agents_from_center,
        center_from_agents = center_from_agents,
        raster_from_world=raster_from_world,
        agent_from_world=batch["centered_agent_from_world_tf"],
        world_from_agent=centered_world_from_agent,
    )

    return d 

def parse_node_centric(batch: dict,rasterize_mode:str):
    maybe_pad_neighbor(batch)
    fut_pos, fut_yaw, _, fut_mask = trajdata2posyawspeed(batch["agent_fut"])
    hist_pos, hist_yaw, hist_speed, hist_mask = trajdata2posyawspeed(batch["agent_hist"])
    curr_speed = hist_speed[..., -1]
    curr_state = batch["curr_agent_state"]
    assert isinstance(curr_state,StateTensor) or isinstance(curr_state,StateArray)
    curr_yaw = curr_state.heading[...,0]
    curr_pos = curr_state.position

    # convert nuscenes types to l5kit types
    agent_type = batch["agent_type"]
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch["agent_hist_extent"]
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    neigh_hist_pos, neigh_hist_yaw, neigh_hist_speed, neigh_hist_mask = trajdata2posyawspeed(batch["neigh_hist"])
    neigh_fut_pos, neigh_fut_yaw, _, neigh_fut_mask = trajdata2posyawspeed(batch["neigh_fut"])
    neigh_curr_speed = neigh_hist_speed[..., -1]
    neigh_types = batch["neigh_types"]
    # convert nuscenes types to l5kit types
    neigh_types[neigh_types < 0] = 0
    neigh_types[neigh_types == 1] = 3
    # mask out invalid extents
    neigh_hist_extents = batch["neigh_hist_extents"]
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch["agents_from_world_tf"])


    # map-related
    if batch["maps"] is not None and batch["maps"].nelement() > 0:
        map_res = batch["maps_resolution"][0]
        h, w = batch["maps"].shape[-2:]
        # TODO: pass env configs to here
        raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(curr_state.device)
        agent_from_raster = torch.inverse(raster_from_agent)
        raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=batch["maps"].shape[0], dim=0)
        agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=batch["maps"].shape[0], dim=0)
        raster_from_world = torch.bmm(raster_from_agent, batch["agents_from_world_tf"])
        all_hist_pos = torch.cat((hist_pos[:, None], neigh_hist_pos), dim=1)
        all_hist_yaw = torch.cat((hist_yaw[:, None], neigh_hist_yaw), dim=1)

        all_extents = torch.cat((batch["agent_hist_extent"].unsqueeze(1),batch["neigh_hist_extents"]),1).max(dim=2)[0][...,:2]
        all_hist_mask = torch.cat((hist_mask[:, None], neigh_hist_mask), dim=1)
        if rasterize_mode=="none":
            maps = batch["maps"]
        elif rasterize_mode=="point":
                maps = rasterize_agents(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        elif rasterize_mode=="square":
            maps = rasterize_agents_rec(
                batch["maps"],
                all_hist_pos,
                all_hist_yaw,
                all_extents,
                all_hist_mask,
                raster_from_agent,
                map_res
            )
        else:
            raise Exception("unknown rasterization mode")
        drivable_map = get_drivable_region_map(batch["maps"])
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0
    d = dict(
        image=maps,
        drivable_map=drivable_map,
        map_names = batch["map_names"],
        target_positions=fut_pos,
        target_yaws=fut_yaw,
        target_availabilities=fut_mask,
        history_positions=hist_pos,
        history_yaws=hist_yaw,
        history_availabilities=hist_mask,
        curr_speed=curr_speed,
        centroid=curr_pos,
        yaw=curr_yaw,
        type=agent_type,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        raster_from_agent=raster_from_agent,
        agent_from_raster=agent_from_raster,
        raster_from_world=raster_from_world,
        agent_from_world=batch["agents_from_world_tf"],
        world_from_agent=world_from_agents,
        all_other_agents_history_positions=neigh_hist_pos,
        all_other_agents_history_yaws=neigh_hist_yaw,
        all_other_agents_history_availability=neigh_hist_mask,
        all_other_agents_history_availabilities=neigh_hist_mask,  # dump hack to agree with l5kit's typo ...
        all_other_agents_curr_speed=neigh_curr_speed,
        all_other_agents_future_positions=neigh_fut_pos,
        all_other_agents_future_yaws=neigh_fut_yaw,
        all_other_agents_future_availability=neigh_fut_mask,
        all_other_agents_types=neigh_types,
        all_other_agents_extents=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
        all_other_agents_history_extents=neigh_hist_extents * extent_scale,
    )
    if "agent_lanes" in batch:
        d["ego_lanes"] = batch["agent_lanes"]
    
    return d

@torch.no_grad()
def parse_trajdata_batch(batch: dict,rasterize_mode="point"):
    
    if "num_agents" in batch:
        # scene centric
        d = parse_scene_centric(batch,rasterize_mode)
        
    else:
        # agent centric
        d = parse_node_centric(batch,rasterize_mode)

    batch = dict(batch)
    batch.update(d)
    for k,v in batch.items():
        if isinstance(v,torch.Tensor):
            batch[k]=v.nan_to_num(0)
    batch.pop("agent_name", None)
    batch.pop("robot_fut", None)
    return batch


def get_modality_shapes(cfg: ExperimentConfig,rasterize_mode:str="point"):
    h = cfg.env.rasterizer.raster_size
    if rasterize_mode=="none":
        return dict(static=(3,h,h),dynamic=(0,h,h),image=(3,h,h))
    else:
        num_channels = (cfg.algo.history_num_frames + 1) + 3
        return dict(static=(3,h,h),dynamic=(cfg.algo.history_num_frames + 1,h,h),image=(num_channels, h, h))
        

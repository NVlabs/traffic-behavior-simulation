from collections import defaultdict
import pdb
import numpy as np
from scipy.interpolate import interp1d
import torch

from tbsim.models.cnn_roi_encoder import rasterized_ROI_align
from tbsim.utils.batch_utils import batch_utils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.tensor_utils import unsqueeze
import tbsim.utils.tensor_utils as TensorUtils
TRAJ_INDEX = [0, 1, 4]  
from tbsim.utils.tree import Tree
from Pplan.Sampling.trajectory_tree import TrajTree
class AgentTrajTree(Tree):
    def __init__(self, traj, parent, depth, prob=None):
        self.traj = traj
        self.children = list()
        self.parent = parent
        if parent is not None:
            parent.expand(self)
        self.depth = depth
        self.prob = prob
        self.attribute = dict()

# The state in Pplan contains more higher orderƒ derivatives, TRAJ_INDEX selects x,y, and heading 
# out of the longer state vector



def get_collision_loss(
    ego_trajectories,
    agent_trajectories,
    ego_extents,
    agent_extents,
    raw_types,
    prob=None,
    col_funcs=None,
):
    """Get veh-veh and veh-ped collision loss."""
    # with torch.no_grad():
    ego_edges, type_mask = batch_utils().gen_ego_edges(
        ego_trajectories, agent_trajectories, ego_extents, agent_extents, raw_types
    )
    if col_funcs is None:
        col_funcs = {
            "VV": GeoUtils.VEH_VEH_collision,
            "VP": GeoUtils.VEH_PED_collision,
        }
    B, N, T = ego_trajectories.shape[:3]
    col_loss = torch.zeros([B, N]).to(ego_trajectories.device)
    for et, func in col_funcs.items():
        dis = func(ego_edges[..., 0:3],ego_edges[..., 3:6],ego_edges[..., 6:8],ego_edges[..., 8:])
        if dis.nelement() > 0:
            col_loss += torch.sum(torch.sigmoid(-dis*4) * type_mask[et][:,None,:,None], dim=[2,3])/T
    
    return col_loss


def get_drivable_area_loss(
    ego_trajectories, raster_from_agent, dis_map, ego_extents
):
    """Cost for road departure."""
    with torch.no_grad():

        lane_flags = rasterized_ROI_align(
            dis_map,
            ego_trajectories[..., :2],
            ego_trajectories[..., 2:],
            raster_from_agent,
            torch.ones(*ego_trajectories.shape[:3]
                       ).to(ego_trajectories.device),
            ego_extents.unsqueeze(1).repeat(1, ego_trajectories.shape[1], 1),
            1,
        ).squeeze(-1)
    return lane_flags.max(dim=-1)[0]

def get_lane_loss_vectorized(ego_trajectories, lane_info, ego_extents):
    Ne,T = ego_trajectories.shape[:2]
    cost = torch.zeros(Ne,device=ego_trajectories.device)

    if "leftbdry" in lane_info:
        delta_x,delta_y,_ = GeoUtils.batch_proj(ego_trajectories.reshape(-1,3),TensorUtils.to_torch(lane_info["leftbdry"],device=ego_extents.device)[None].repeat_interleave(Ne*T,0))
        leftmargin = -delta_y.max(1)[0].reshape(Ne,T)-ego_extents[1]/2
        cost += -(leftmargin.min(1)[0]).clamp(max=0)
    if "rightbdry" in lane_info:
        delta_x,delta_y,_ = GeoUtils.batch_proj(ego_trajectories.reshape(-1,3),TensorUtils.to_torch(lane_info["rightbdry"],device=ego_extents.device)[None].repeat_interleave(Ne*T,0))
        rightmargin = delta_y.min(1)[0].reshape(Ne,T)-ego_extents[1]/2
        cost += -(rightmargin.min(1)[0]).clamp(max=0)
    return cost

def get_lane_loss_simple(ego_trajectories, raster_from_agent, dis_map):
    h,w = dis_map.shape[-2:]
    
    raster_xy = GeoUtils.batch_nd_transform_points(ego_trajectories[...,:2],raster_from_agent)
    raster_xy[...,0] = raster_xy[...,0].clip(0,w-1e-5)
    raster_xy[...,1] = raster_xy[...,1].clip(0,h-1e-5)
    raster_xy = raster_xy.long()
    raster_xy_flat = (raster_xy[...,1]*w+raster_xy[...,0])
    raster_xy_flat = raster_xy_flat.flatten()
    lane_loss = (dis_map.flatten()[raster_xy_flat]).reshape(*raster_xy.shape[:2])
    return lane_loss.max(dim=-1)[0]

def get_terminal_likelihood_reward(
    ego_trajectories, raster_from_agent, log_likelihood
):
    """Cost for road departure."""

    log_likelihood = (log_likelihood-log_likelihood.mean())/log_likelihood.std()
    h,w = log_likelihood.shape[-2:]
    
    raster_xy = GeoUtils.batch_nd_transform_points(ego_trajectories[...,-1,:2],raster_from_agent)
    raster_xy[...,0] = raster_xy[...,0].clip(0,w-1e-5)
    raster_xy[...,1] = raster_xy[...,1].clip(0,h-1e-5)
    raster_xy = raster_xy.long()
    raster_xy_flat = (raster_xy[...,1]*w+raster_xy[...,0])

    ll_reward = log_likelihood.flatten()[raster_xy_flat]
    return ll_reward

def get_progress_reward(ego_trajectories,d_sat = 10):
    dis = torch.linalg.norm(ego_trajectories[...,-1,:2]-ego_trajectories[...,0,:2],dim=-1)
    return 2/np.pi*torch.atan(dis/d_sat)


def get_total_distance(ego_trajectories):
    """Reward that incentivizes progress."""
    # Assume format [..., T, 3]
    assert ego_trajectories.shape[-1] == 3
    diff = ego_trajectories[..., 1:, :] - ego_trajectories[..., :-1, :]
    dist = torch.norm(diff[..., :2], dim=-1)
    total_dist = torch.sum(dist, dim=-1)
    return total_dist


def ego_sample_planning(
        ego_trajectories,
        agent_trajectories,
        ego_extents,
        agent_extents,
        raw_types,
        raster_from_agent,
        dis_map,
        weights,
        vector_map=None,
        world_from_agent=None,
        yaw = None,
        log_likelihood=None,
        col_funcs=None,
):
    """A basic cost function for prediction-and-planning"""
    col_loss = get_collision_loss(
        ego_trajectories,
        agent_trajectories,
        ego_extents,
        agent_extents,
        raw_types,
        col_funcs,
    )
    lane_loss = get_drivable_area_loss(
        ego_trajectories, raster_from_agent, dis_map, ego_extents
    )

    progress = get_total_distance(ego_trajectories)

    log_likelihood = 0 if log_likelihood is None else log_likelihood
    if log_likelihood.ndim==3:
        log_likelihood = get_terminal_likelihood_reward(ego_trajectories, raster_from_agent, log_likelihood)
    total_score = (
            + weights["likelihood_weight"] * log_likelihood
            + weights["progress_weight"] * progress
            - weights["collision_weight"] * col_loss
            - weights["lane_weight"] * lane_loss
    )
    if vector_map is not None and world_from_agent is not None and yaw is not None:

        world_pos = GeoUtils.batch_nd_transform_points(ego_trajectories[...,:2],world_from_agent[:,None,None])
        world_yaw = ego_trajectories[...,2]+yaw[:,None,None]
        world_pos,world_yaw = TensorUtils.to_numpy((world_pos,world_yaw))
        world_xyz = np.concatenate([world_pos,np.zeros_like(world_pos[...,0:1])],axis=-1)
        world_xyh = np.concatenate([world_pos,world_yaw[...,None]],axis=-1)
        unique_ids = set([vmap.map_id for vmap in vector_map])
        bs,Ne = ego_trajectories.shape[:2]
        heading_error = np.inf*np.ones([bs,Ne])
        depth = 3
        min_dis = np.inf*np.ones([bs,Ne])
        for i,vmap in enumerate(vector_map):
            
            nearby_lanes = vmap.get_lanes_within(world_xyz[i,0,0],30)
            dis = np.inf*np.ones([len(nearby_lanes)])
            for j,lane in enumerate(nearby_lanes):
                dx,dy,dh = GeoUtils.batch_proj(world_xyh[i,0,0],lane.center.points[None,...,[0,1,3]])
                pts_idx = np.abs(dx).argmin(-1)
                dy_min = np.abs(dy[0,pts_idx])
                if (dx.min(-1)<0) & (dx.max(-1)>0) & (dy_min<2) & (np.abs(dh)<np.pi/4):
                    dis[j] = dy_min+5*np.abs(dh)
                else:
                    continue
            current_lane = nearby_lanes[np.argmin(dis)]
            relevant_lanes = [{current_lane}]
            for d in range(depth):
                next_lanes = set()
                for lane in relevant_lanes[d]:
                    next_lanes.update({vmap.get_road_lane(id) for id in lane.next_lanes})
                relevant_lanes.append(next_lanes)
            for d in range(depth):
                relevant_lanes[depth].update(relevant_lanes[d])
            relevant_lanes = relevant_lanes[depth]
            
            for lane in relevant_lanes:
                dx,dy,dh = GeoUtils.batch_proj(world_xyh[i,:,-1],lane.center.points[None,...,[0,1,3]].repeat(Ne,0))
                dh = np.abs(dh).squeeze(-1)
                pts_idx = np.abs(dx).argmin(-1)
                dy_min = np.abs(np.take_along_axis(dy,pts_idx[...,None],-1).squeeze(-1))
                
                onroute_flag = (dx.min(-1)<0) & (dx.max(-1)>0) & (dy_min<2) & (dh<np.pi/4)
                dis = dy_min+5*dh
                dis[~onroute_flag] = np.inf
                min_dis[i] = np.minimum(min_dis[i],dis)
        lane_dir_loss = torch.from_numpy(min_dis).to(ego_trajectories.device)
        lane_dir_loss = torch.clip(lane_dir_loss,min=0,max=10)
        total_score-=weights["lane_dir_weight"]*lane_dir_loss

    return torch.argmax(total_score, dim=1)


class TreeMotionPolicy(object):
    """ A trajectory tree policy as the result of contingency planning

    """
    def __init__(self,stage,num_frames_per_stage,ego_root,scenario_root,cost_to_go,leaf_idx,curr_node):
        self.stage = stage
        self.num_frames_per_stage = num_frames_per_stage
        self.ego_root = ego_root
        self.scenario_root = scenario_root
        self.cost_to_go = cost_to_go
        self.leaf_idx = leaf_idx
        self.curr_node= curr_node

    def identify_branch(self,ego_node,scene_traj):

        assert scene_traj.shape[-2]<self.stage*self.num_frames_per_stage
        assert ego_node.total_traj.shape[0]-1>=scene_traj.shape[-2]

        remain_traj = scene_traj
        curr_scenario_node = self.scenario_root
        ego_leaf_index = self.leaf_idx[ego_node]
        while remain_traj.shape[1]>0:
            seg_length = min(remain_traj.shape[-2],self.num_frames_per_stage)
            dis = [torch.linalg.norm(child.traj[ego_leaf_index,:,:seg_length,:2]-remain_traj[:,:seg_length,:2],dim=-1).sum().item()\
                    for child in curr_scenario_node.children]
            idx = torch.argmin(torch.tensor(dis)).item()

            curr_scenario_node = curr_scenario_node.children[idx]
            
            remain_traj = remain_traj[...,seg_length:,:]
            remain_num_frames = curr_scenario_node.traj.shape[-2]-seg_length
            if curr_scenario_node.stage>=self.curr_node.stage:
                break
        return curr_scenario_node

    def get_plan(self,scene_traj,horizon):
        if scene_traj is None:
            T = 0
            remain_num_frames = self.num_frames_per_stage
        else:
            T = scene_traj.shape[-2]
            remain_num_frames = self.curr_node.total_traj.shape[0]-1-T
            assert remain_num_frames>-self.num_frames_per_stage
            if remain_num_frames<=0:
                assert not self.curr_node.isleaf()
                curr_scenario_node = self.identify_branch(self.curr_node,scene_traj)
                Q = [self.cost_to_go[(child,curr_scenario_node)] for child in self.curr_node.children]
                idx = torch.argmin(torch.tensor(Q)).item()
                self.curr_node = self.curr_node.children[idx]
                remain_num_frames+=self.curr_node.traj.shape[0]
        traj = self.curr_node.traj[-remain_num_frames:,TRAJ_INDEX]
        if not self.curr_node.isleaf():
            traj = torch.cat((traj,self.curr_node.children[0].traj[:,TRAJ_INDEX]),-2)
        if traj.shape[0]>=horizon:
            return traj[:horizon]
        else:
            traj_patched = torch.cat((traj,traj[-1].tile(horizon-traj.shape[0],1)))
            return traj_patched

class VectorizedTreeMotionPolicy(TreeMotionPolicy):
    """ A vectorized trajectory tree policy as the result of contingency planning

    """
    def __init__(self,stage,num_frames_per_stage,ego_tree,children_indices,scenario_tree,cost_to_go,leaf_idx,curr_node):
        self.stage = stage
        self.num_frames_per_stage = num_frames_per_stage
        self.ego_tree = ego_tree
        self.ego_root = ego_tree[0][0]
        self.children_indices = children_indices
        self.scenario_tree = scenario_tree
        self.scenario_root = scenario_tree[0][0]
        self.cost_to_go = cost_to_go
        self.leaf_idx = leaf_idx
        self.curr_node= curr_node

    def identify_branch(self,ego_node,scene_traj):

        assert scene_traj.shape[-2]<self.stage*self.num_frames_per_stage
        assert ego_node.total_traj.shape[0]-1>=scene_traj.shape[-2]

        remain_traj = scene_traj
        curr_scenario_node = self.scenario_root
        
        stage = ego_node.depth
        ego_stage_index = self.ego_tree[stage].index(ego_node)
        ego_leaf_index = self.leaf_idx[stage][ego_stage_index].item()
        while remain_traj.shape[1]>0:
            seg_length = min(remain_traj.shape[-2],self.num_frames_per_stage)
            dis = [torch.linalg.norm(child.traj[ego_leaf_index,:,:seg_length,:2]-remain_traj[:,:seg_length,:2],dim=-1).sum().item()\
                    for child in curr_scenario_node.children]
            idx = torch.argmin(torch.tensor(dis)).item()

            curr_scenario_node = curr_scenario_node.children[idx]
            
            remain_traj = remain_traj[...,seg_length:,:]
            remain_num_frames = curr_scenario_node.traj.shape[-2]-seg_length
            if curr_scenario_node.stage>=self.curr_node.stage:
                break
        return curr_scenario_node

    def get_plan(self,scene_traj,horizon):
        if scene_traj is None:
            T = 0
            remain_num_frames = self.num_frames_per_stage
        else:
            T = scene_traj.shape[-2]
            remain_num_frames = self.curr_node.total_traj.shape[0]-1-T
            assert remain_num_frames>-self.num_frames_per_stage
            if remain_num_frames<=0:
                assert not self.curr_node.isleaf()
                curr_scenario_node = self.identify_branch(self.curr_node,scene_traj)
                assert curr_scenario_node.depth==self.curr_node.depth
                stage = self.curr_node.depth
                scene_node_idx = self.scenario_tree[stage].index(curr_scenario_node)
                curr_node_idx = self.ego_tree[stage].index(self.curr_node)
                Q = self.cost_to_go[stage][scene_node_idx,self.children_indices[stage][curr_node_idx]]
                idx = torch.argmin(Q).item()
                self.curr_node = self.curr_node.children[idx]
                remain_num_frames+=self.curr_node.traj.shape[0]

        traj = self.curr_node.traj[-remain_num_frames:,TRAJ_INDEX]
        if not self.curr_node.isleaf():
            traj = torch.cat((traj,self.curr_node.children[0].traj[:,TRAJ_INDEX]),-2)
        if traj.shape[0]>=horizon:
            return traj[:horizon]
        else:
            traj_patched = torch.cat((traj,traj[-1].tile(horizon-traj.shape[0],1)))
            return traj_patched
            

def tiled_to_tree(total_traj,prob,num_stage,num_frames_per_stage,M):
    """Turning a trajectory tree in tiled form to a tree data structure

    Args:
        total_traj (torch.tensor or np.ndarray): tiled trajectory tree
        prob (torch.tensor or np.ndarray): probability of the modes
        num_stage (int): number of layers of the tree
        num_frames_per_stage (int): number of time frames per layer
        M (int): branching factor

    Returns:
        nodes (dict[int:List(AgentTrajTree)]): all branches of the trajectory tree nodes indexed by layer
    """

    # total_traj = TensorUtils.reshape_dimensions_single(total_traj,2,3,[M]*num_stage)
    x0 = AgentTrajTree(None, None, 0)
    nodes = defaultdict(lambda:list())
    nodes[0].append(x0)
    for t in range(num_stage):
        interval = M**(num_stage-t-1)
        tiled_traj = total_traj[...,::interval,:,t*num_frames_per_stage:(t+1)*num_frames_per_stage,:]
        for i in range(M**(t+1)):
            parent_idx = int(i/M)
            p = prob[:,i*interval:(i+1)*interval].sum(-1)
            node = AgentTrajTree(tiled_traj[:,i], nodes[t][parent_idx], t + 1, prob=p)
            nodes[t+1].append(node)
    return nodes


def contingency_planning(ego_tree,
                         ego_extents,
                         agent_traj,
                         mode_prob,
                         agent_extents,
                         agent_types,
                         raster_from_agent,
                         dis_map,
                         weights,
                         num_frames_per_stage,
                         M,
                         dt,
                         col_funcs=None,
                         log_likelihood=None,
                         pert_std = None):
    """A sampling-based contingency planning algorithm

    Args:
        ego_tree (Dict[int:List[TrajTree]]): ego trajectory tree
        ego_extents (Tensor): [2]
        agent_traj (Tensor): EC x M^s x Na x (s*Ts) x 3 scenario tree predicted
        mode_prob (Tensor): EC x M^s
        agent_extents (Tensor): Na x 2
        agent_types (Tensor): Na
        raster_from_agent (Tensor): 3 x 3
        dis_map (Tensor): 224 x 224 (same as rasterized feature dimension) distance map
        weights (Dict): weights of various costs
        num_frames_per_stage (int): Ts
        M (int): branching factor
        col_funcs (function handle, optional): in case custom collision function is used. Defaults to None.

    Returns:
        TreeMotionPolicy: optimal motion policy
    """
    
    num_stage = len(ego_tree)-1
    ego_root = ego_tree[0][0]
    device = agent_traj.device
    leaf_idx = defaultdict(lambda:list())
    for stage in range(num_stage,-1,-1):
        for node in ego_tree[stage]:
            if node.isleaf():
                leaf_idx[node]=[ego_tree[stage].index(node)]
            else:
                leaf_idx[node] = []
                for child in node.children:
                    leaf_idx[node] = leaf_idx[node]+leaf_idx[child]
    

    V = dict()
    L = dict()
    Q = dict()
    scenario_tree = tiled_to_tree(agent_traj,mode_prob,num_stage,num_frames_per_stage,M)
    scenario_root = scenario_tree[0][0]
    v0 = ego_root.traj[0,2]
    d_sat = v0.clip(min=2.0)*num_frames_per_stage*dt
    for stage in range(num_stage,0,-1):
        if stage==0:
            total_loss = torch.zeros([1,1],device=device)
        else:
            ego_nodes = ego_tree[stage]
            indices = [leaf_idx[node][0] for node in ego_nodes]
            ego_traj = [node.traj[:,TRAJ_INDEX] for node in ego_nodes]
            ego_traj = torch.stack(ego_traj,0)
            agent_nodes = scenario_tree[stage]
            agent_traj = [node.traj[indices] for node in agent_nodes]
            agent_traj = torch.stack(agent_traj,0)
            ego_traj_tiled = ego_traj.unsqueeze(0).repeat(len(agent_nodes),1,1,1)
            col_loss = get_collision_loss(ego_traj_tiled,
                                        agent_traj,
                                        ego_extents.tile(len(agent_nodes),1),
                                        agent_extents.tile(len(agent_nodes),1,1),
                                        agent_types.tile(len(agent_nodes),1),
                                        col_funcs,
                                        )


            lane_loss = get_drivable_area_loss(ego_traj.unsqueeze(0), raster_from_agent.unsqueeze(0), dis_map.unsqueeze(0), ego_extents.unsqueeze(0))
            # lane_loss = get_lane_loss_simple(ego_traj,raster_from_agent,dis_map).unsqueeze(0)

            progress_reward = get_progress_reward(ego_traj,d_sat=d_sat)
            
            total_loss = weights["collision_weight"]*col_loss+weights["lane_weight"]*lane_loss-weights["progress_weight"]*progress_reward.unsqueeze(0)
            if pert_std is not None:
                total_loss +=torch.randn(total_loss.shape[1],device=device).unsqueeze(0)*pert_std
            if log_likelihood is not None and stage==num_stage:
                ll_reward = get_terminal_likelihood_reward(ego_traj, raster_from_agent, log_likelihood)
                total_loss = total_loss-weights["likelihood_weight"]*ll_reward
            

        for i in range(len(ego_nodes)):
            for j in range(len(agent_nodes)):
                L[(ego_nodes[i],agent_nodes[j])] = total_loss[j,i]
                if stage==num_stage:
                    V[(ego_nodes[i],agent_nodes[j])] = float(total_loss[j,i])
                else:
                    children_cost_to_go = [Q[(child,agent_nodes[j])] for child in ego_nodes[i].children]
                    V[(ego_nodes[i],agent_nodes[j])] = float(total_loss[j,i])+min(children_cost_to_go)

            if stage>0:
                for agent_node in scenario_tree[stage-1]:
                    cost_i = []
                    prob_i = []
                    for child in agent_node.children:
                        cost_i.append(V[ego_nodes[i],child])
                        prob_i.append(child.prob[leaf_idx[ego_nodes[i]]].sum())
                    cost_i = torch.tensor(cost_i,device=device)
                    prob_i = torch.stack(prob_i)
                    Q[(ego_nodes[i],agent_node)] = float((cost_i*prob_i).sum()/prob_i.sum())
    Q_root = [Q[(child,scenario_root)] for child in ego_root.children]
    idx = torch.argmin(torch.tensor(Q_root)).item()
    optimal_node = ego_root.children[idx]
    motion_policy = TreeMotionPolicy(num_stage,
                                     num_frames_per_stage,
                                     ego_root,
                                     scenario_root,
                                     Q,
                                     leaf_idx,
                                     optimal_node)
    motion_policy.get_plan(None,num_stage*num_frames_per_stage)
    return motion_policy

def contingency_planning_parallel(ego_tree,
                         ego_extents,
                         agent_traj,
                         mode_prob,
                         agent_extents,
                         agent_types,
                         raster_from_agent,
                         lane_info,
                         weights,
                         num_frames_per_stage,
                         M,
                         dt,
                         col_funcs=None,
                         log_likelihood=None,
                         pert_std = None,
                         lane_type="rasterized"):
    """A sampling-based contingency planning algorithm

    Args:
        ego_tree (Dict[int:List[TrajTree]]): ego trajectory tree
        ego_extents (Tensor): [2]
        agent_traj (Tensor): EC x M^s x Na x (s*Ts) x 3 scenario tree predicted
        mode_prob (Tensor): EC x M^s
        agent_extents (Tensor): Na x 2
        agent_types (Tensor): Na
        raster_from_agent (Tensor): 3 x 3
        lane_info: either a tensor of 224 x 224 (same as rasterized feature dimension) distance map or map api
        weights (Dict): weights of various costs
        num_frames_per_stage (int): Ts
        M (int): branching factor
        col_funcs (function handle, optional): in case custom collision function is used. Defaults to None.

    Returns:
        TreeMotionPolicy: optimal motion policy
    """
    device=agent_traj.device
    children_indices = TrajTree.get_children_index_torch(ego_tree)
    num_stage = len(ego_tree)-1
    ego_root = ego_tree[0][0]

    leaf_idx = {num_stage:torch.arange(len(ego_tree[num_stage]),device=device)}
    stage_prob = {num_stage:mode_prob.T}
    for stage in range(num_stage-1,-1,-1):
        leaf_idx[stage] = leaf_idx[stage+1][children_indices[stage][:,0]]
        prob_next = stage_prob[stage+1]
        stage_prob[stage] = prob_next.reshape(-1,M,prob_next.shape[-1])[:,:,children_indices[stage][:,0]].sum(1)

    V = dict()
    L = dict()
    Q = dict()


    scenario_tree = tiled_to_tree(agent_traj,mode_prob,num_stage,num_frames_per_stage,M)
    scenario_root = scenario_tree[0][0]
    v0 = ego_root.traj[0,2]
    d_sat = v0.clip(min=2.0)*num_frames_per_stage*dt

    
    
    for stage in range(num_stage,-1,-1):
        if stage==0:
            total_loss = torch.zeros([1,1],device=device)
        else:
            #calculate stage cost
            ego_nodes = ego_tree[stage]
            ego_traj = [node.traj[:,TRAJ_INDEX] for node in ego_nodes]
            ego_traj = torch.stack(ego_traj,0)
            agent_nodes = scenario_tree[stage]
            
            agent_traj = [node.traj[leaf_idx[stage]] for node in agent_nodes]

            agent_traj = torch.stack(agent_traj,0)
            ego_traj_tiled = ego_traj.unsqueeze(0).repeat(len(agent_nodes),1,1,1)
            col_loss = get_collision_loss(ego_traj_tiled,
                                        agent_traj,
                                        ego_extents.tile(len(agent_nodes),1),
                                        agent_extents.tile(len(agent_nodes),1,1),
                                        agent_types.tile(len(agent_nodes),1),
                                        col_funcs,
                                        )

            if lane_type=="rasterized":
                lane_loss = get_drivable_area_loss(ego_traj.unsqueeze(0), raster_from_agent.unsqueeze(0), lane_info.unsqueeze(0), ego_extents.unsqueeze(0))
            else:
                lane_loss = get_lane_loss_vectorized(ego_traj,lane_info,ego_extents).unsqueeze(0)
            # lane_loss = get_lane_loss_simple(ego_traj,raster_from_agent,dis_map).unsqueeze(0)

            progress_reward = get_progress_reward(ego_traj,d_sat=d_sat)
            
            total_loss = weights["collision_weight"]*col_loss+weights["lane_weight"]*lane_loss-weights["progress_weight"]*progress_reward.unsqueeze(0)
            if pert_std is not None:
                total_loss +=torch.randn(total_loss.shape[1],device=device).unsqueeze(0)*pert_std
            if log_likelihood is not None and stage==num_stage:
                ll_reward = get_terminal_likelihood_reward(ego_traj, raster_from_agent, log_likelihood)
                total_loss = total_loss-weights["likelihood_weight"]*ll_reward
            
        L[stage] = total_loss
        if stage==num_stage:
            V[stage] = total_loss
        else:
            children_idx = children_indices[stage]
            # add the last Q value as inf since empty children index are padded with -1
            Q_prime = torch.cat((Q[stage],torch.full([Q[stage].shape[0],1],np.inf,device=device)),1)
            Q_by_node = Q_prime[:,children_idx]
            V[stage] = total_loss+Q_by_node.min(dim=-1)[0]
        
        if stage>0:
            children_V = V[stage]
            children_V = children_V.reshape(-1,M,children_V.shape[-1])
            prob = stage_prob[stage]
            prob = prob.reshape(-1,M,prob.shape[-1])
            prob_normalized = prob/prob.sum(dim=1,keepdim=True)
            Q[stage-1] = (children_V*prob_normalized).sum(dim=1)

    idx = Q[0].argmin().item()

    motion_policy = VectorizedTreeMotionPolicy(num_stage,
                                               num_frames_per_stage,
                                               ego_tree,
                                               children_indices,
                                               scenario_tree,
                                               Q,
                                               leaf_idx,
                                               ego_root.children[idx])
    return motion_policy

def one_shot_planning(ego_tree,
                         ego_extents,
                         agent_traj,
                         mode_prob,
                         agent_extents,
                         agent_types,
                         raster_from_agent,
                         dis_map,
                         weights,
                         num_frames_per_stage,
                         M,
                         dt,
                         col_funcs=None,
                         log_likelihood=None,
                         pert_std = None,
                         strategy="all"):

    """Alternative of contingency planning, try to avoid all predicted trajectories

    Args:
        ego_tree (Dict[int:List[TrajTree]]): ego trajectory tree
        ego_extents (Tensor): [2]
        agent_traj (Tensor): EC x M^s x Na x (s*Ts) x 3 scenario tree predicted
        mode_prob (Tensor): EC x M^s
        agent_extents (Tensor): Na x 2
        agent_types (Tensor): Na
        raster_from_agent (Tensor): 3 x 3
        dis_map (Tensor): 224 x 224 (same as rasterized feature dimension) distance map
        weights (Dict): weights of various costs
        num_frames_per_stage (int): Ts
        M (int): branching factor
        col_funcs (function handle, optional): in case custom collision function is used. Defaults to None.

    Returns:
        TreeMotionPolicy: optimal motion policy
    """
    assert strategy=="all" or strategy=="maximum"
    num_stage = len(ego_tree)-1
    ego_root = ego_tree[0][0]
    
    ego_traj = [node.total_traj[1:,TRAJ_INDEX] for node in ego_tree[num_stage]]
    ego_traj = torch.stack(ego_traj,0)
    ego_traj_tiled = ego_traj.unsqueeze(1).repeat_interleave(agent_traj.shape[1],1)
    Ne = ego_traj.shape[0]
    if strategy=="maximum":
        idx = mode_prob.argmax(dim=1)
        idx = idx.reshape(Ne,*[1]*(agent_traj.ndim-1))
        agent_traj = agent_traj.take_along_dim(idx,1)
    col_loss = get_collision_loss(ego_traj_tiled,agent_traj,ego_extents.tile(Ne,1),agent_extents.tile(Ne,1,1),agent_types.tile(Ne,1),col_funcs)
    col_loss = col_loss.max(dim=1)[0]
    lane_loss = get_drivable_area_loss(ego_traj.unsqueeze(0), raster_from_agent.unsqueeze(0), dis_map.unsqueeze(0), ego_extents.unsqueeze(0)).squeeze(0)
    v0 = ego_root.traj[0,2]
    d_sat = v0.clip(min=2.0)*num_frames_per_stage*dt
    progress_reward = get_progress_reward(ego_traj,d_sat=d_sat)
    total_loss = weights["collision_weight"]*col_loss+weights["lane_weight"]*lane_loss-weights["progress_weight"]*progress_reward
    if pert_std is not None:
        total_loss +=torch.randn(total_loss.shape[0],device=total_loss.device)*pert_std
    if log_likelihood is not None:
        ll_reward = get_terminal_likelihood_reward(ego_traj, raster_from_agent, log_likelihood)
        total_loss = total_loss-weights["likelihood_weight"]*ll_reward

    idx = total_loss.argmin()
    return ego_traj[idx]

            
def obtain_ref(line, x, v, N, dt):
    """obtain desired trajectory for the MPC controller

    Args:
        line (np.ndarray): centerline of the lane [n, 3]
        x (np.ndarray): position of the vehicle
        v (np.ndarray): desired velocity
        N (int): number of time steps
        dt (float): time step

    Returns:
        refx (np.ndarray): desired trajectory [N,3]
    """
    line_length = line.shape[0]
    delta_x = line[..., 0:2] - np.repeat(x[..., np.newaxis, 0:2], line_length, axis=-2)
    dis = np.linalg.norm(delta_x, axis=-1)
    idx = np.argmin(dis, axis=-1)
    line_min = line[idx]
    dx = x[0] - line_min[0]
    dy = x[1] - line_min[1]
    delta_y = -dx * np.sin(line_min[2]) + dy * np.cos(line_min[2])
    delta_x = dx * np.cos(line_min[2]) + dy * np.sin(line_min[2])
    refx0 = np.array(
        [
            line_min[0] + delta_x * np.cos(line_min[2]),
            line_min[1] + delta_x * np.sin(line_min[2]),
            line_min[2],
        ]
    )
    s = [np.linalg.norm(line[idx + 1, 0:2] - refx0[0:2])]
    for i in range(idx + 2, line_length):
        s.append(s[-1] + np.linalg.norm(line[i, 0:2] - line[i - 1, 0:2]))
    f = interp1d(
        np.array(s),
        line[idx + 1 :],
        kind="linear",
        axis=0,
        copy=True,
        bounds_error=False,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    s1 = v * np.arange(1, N + 1) * dt
    refx = f(s1)

    return refx
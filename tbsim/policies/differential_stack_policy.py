import enum
from glob import glob
from os import device_encoding
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Tuple, Dict
from copy import deepcopy

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.dynamics as dynamics
from tbsim.utils.l5_utils import get_current_states, get_drivable_region_map
from tbsim.algos.algo_utils import optimize_trajectories
from tbsim.utils.geometry_utils import transform_points_tensor, calc_distance_map
from l5kit.geometry import transform_points
from tbsim.utils.timer import Timers
from tbsim.utils.planning_utils import ego_sample_planning
from tbsim.policies.common import Action, Plan
from tbsim.policies.base import Policy
from tbsim.utils.ftocp import FTOCP


try:
    from Pplan.Sampling.spline_planner import SplinePlanner
    from Pplan.Sampling.trajectory_tree import TrajTree
except ImportError:
    print("Cannot import Pplan")

import tbsim.utils.planning_utils as PlanUtils
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.timer import Timers

import sys
# sys.path.append('./trajectron/trajectron')
# sys.path.append('./trajdata/src/trajdata')

from torch import nn, optim
from torch.utils import data
import os
import time
import dill, pickle
import json
import shutil
import random
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt


from diffstack.trajectron.trajectron.model.trajectron import Trajectron
from diffstack.trajectron.trajectron.model.planning_aware_trajectron import PlanningAwareTrajectron
from diffstack.trajectron.trajectron.model.model_registrar import ModelRegistrar
from diffstack.trajectron.trajectron.model.dataset import EnvironmentDataset, collate
from diffstack.trajectron.trajectron.environment.environment import EnvironmentMetadata
from torch.utils.tensorboard import SummaryWriter
# torch.autograd.set_detect_anomaly(True)

import sys
sys.path.append('./Trajectron-plus-plus/trajectron')
import gpu_affinity
import wandb

from diffstack.trajectron.trajectron.utils.comm import all_gather
from collections import defaultdict
from pathlib import Path

import ipdb as pdb

from trajdata import UnifiedDataset, AgentType
from diffstack.trajectron.trajectron.utils.batch_utils import reformat_batch

from diffstack.trajectron.trajectron.model.dataset.preprocessing import get_node_timestep_data, get_timesteps_data
import pandas as pd
from diffstack.trajectron.trajectron.environment import Environment, Scene, Node, GeometricMap, derivative_of
from diffstack.trajectron.experiments.nuScenes.process_plan_data import get_lane_reference_points, CONTROL_FIT_H, data_columns_vehicle
from diffstack.pred_metric.environment import utils as pred_metric_utils
from diffstack.diffmpc.mpc import util as mpc_util
import torch.distributed as dist

def get_hyperparams(diffstack_args):
    # Load hyperparameters from json
    if not os.path.exists(diffstack_args.conf):
        print('Config json not found!')
    with open(diffstack_args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = diffstack_args.dynamic_edges
    hyperparams['edge_state_combine_method'] = diffstack_args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = diffstack_args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = diffstack_args.edge_addition_filter
    hyperparams['edge_removal_filter'] = diffstack_args.edge_removal_filter
    hyperparams['batch_size'] = diffstack_args.batch_size
    hyperparams['k_eval'] = diffstack_args.k_eval
    hyperparams['offline_scene_graph'] = diffstack_args.offline_scene_graph
    hyperparams['incl_robot_node'] = diffstack_args.incl_robot_node
    hyperparams['node_freq_mult_train'] = diffstack_args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = diffstack_args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = diffstack_args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = diffstack_args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = diffstack_args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not diffstack_args.no_edge_encoding
    hyperparams['use_map_encoding'] = diffstack_args.map_encoding
    hyperparams['augment'] = diffstack_args.augment
    hyperparams['override_attention_radius'] = diffstack_args.override_attention_radius
    hyperparams['load'] = diffstack_args.load
    hyperparams['load2'] = diffstack_args.load2
    hyperparams['pred_loss_scaler'] = diffstack_args.pred_loss_scaler
    hyperparams['pred_loss_weights'] = diffstack_args.pred_loss_weights
    hyperparams['pred_loss_temp'] = diffstack_args.pred_loss_temp
    hyperparams['plan_loss_scaler'] = diffstack_args.plan_loss_scaler
    hyperparams['plan_loss_scale_start'] = diffstack_args.plan_loss_scale_start
    hyperparams['plan_loss_scale_end'] = diffstack_args.plan_loss_scale_end
    hyperparams['plan_loss'] = diffstack_args.plan_loss
    hyperparams['plan_init'] = diffstack_args.plan_init
    hyperparams['bias_predictions'] = diffstack_args.bias_predictions
    hyperparams['plan_lqr_eps'] = diffstack_args.plan_lqr_eps
    hyperparams['plan_dt'] = (diffstack_args.plan_dt if diffstack_args.plan_dt > 0. else hyperparams['dt'])

    if diffstack_args.no_plan_train:
        hyperparams['plan_train'] = False
    if diffstack_args.no_train_pred:
        hyperparams['train_pred'] = False
    if diffstack_args.train_plan_cost:
        hyperparams['train_plan_cost'] = True
    if diffstack_args.plan_cost != "":
        hyperparams['plan_cost'] = diffstack_args.plan_cost
    if diffstack_args.plan_cost_for_gt != "":
        hyperparams['plan_cost_for_gt'] = diffstack_args.plan_cost_for_gt
    if diffstack_args.dataset_version != "":
        hyperparams['dataset_version'] = diffstack_args.dataset_version
    if diffstack_args.planner != "":
        hyperparams['planner'] = diffstack_args.planner

    # Fill in defaults for plannig related hyperparams
    for k, v in {
        "all_futures": False,
        "pred_ego_indicator": "none",
        "planner": "",
        "plan_train": False,
        "train_pred": True,
        "train_plan_cost": False,
        "plan_cost": "mpc1",
        "plan_node_types": [],
        "plan_cost_lr": -1.0,
        "plan_agent_choice": "most_relevant",
        "filter_plan_valid": False,
        "filter_plan_converged": False,
        "filter_plan_relevant": False,
        "filter_lane_near": False,
        "plan_lqr_max_iters": 5,
        "plan_lqr_max_linesearch_iters": 5,        
        "preplan_lqr_max_iters": 20,
        "preplan_lqr_max_linesearch_iters": 20,
        "plan_cost_for_gt": "mpc1",
        "dataset_version": "v2",
    }.items():
        if k not in hyperparams:
            hyperparams[k] = v

    # Distributed LR Scaling
    if diffstack_args.learning_rate is not None:
        hyperparams['learning_rate'] = diffstack_args.learning_rate
    hyperparams['learning_rate'] *= dist.get_world_size()

    return hyperparams    



class DiffStackPolicy(Policy):
    def __init__(self,diffstack_args,device="cpu"):
        rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        self.device = device
        self.timer = Timers()
        hyperparams = get_hyperparams(diffstack_args)
        
        model_dir = os.path.join(diffstack_args.log_dir,
                                 diffstack_args.log_tag + time.strftime('-%d_%b_%Y_%H_%M_%S', time.localtime()))
        model_registrar = ModelRegistrar(model_dir, diffstack_args.device)
        if diffstack_args.load:
            model_registrar.load_model_from_file(diffstack_args.load)
        trajectron = PlanningAwareTrajectron(model_registrar,
                            hyperparams,
                            None,
                            device)
        raw_data_path = "/home/yuxiaoc/repos/Trajectron-plus-plus/experiments/nuScenes/v1.0-mini"
        from nuscenes.map_expansion.map_api import NuScenesMap, locations as nusc_map_names
        nusc_maps = {map_name: NuScenesMap(raw_data_path, map_name) for map_name in nusc_map_names}
        if "mini" in diffstack_args.eval_data_dict:
            with open("/home/yuxiaoc/repos/planning-aware-trajectron/diffstack/trajectron/experiments/nuScenes/nusc_mini_sample_scene_map.pkl", "rb") as f:
                self.cached_sample_scene_map = pickle.load(f)
        else:
            with open("/home/yuxiaoc/repos/planning-aware-trajectron/diffstack/trajectron/experiments/nuScenes/nusc_sample_scene_map.pkl", "rb") as f:
                self.cached_sample_scene_map = pickle.load(f)
        cache_params = ".".join([str(hyperparams[k]) for k in 
                            ["prediction_horizon", "plan_cost_for_gt", "preplan_lqr_max_iters", "preplan_lqr_max_linesearch_iters"]] + 
                            [hyperparams["dataset_version"]])
        # cached_train_data_path = os.path.join(diffstack_args.cache_dir, f"{diffstack_args.train_data_dict}.{cache_params}.cached.data.pkl")
        cached_eval_data_path = os.path.join(diffstack_args.cache_dir, f"{diffstack_args.eval_data_dict}.{cache_params}.cached.data.pkl")
        eval_scenes = []
        eval_scenes_sample_probs = None
        eval_data_path = os.path.join(diffstack_args.data_dir, diffstack_args.eval_data_dict)
        
        sys.path.append("/home/yuxiaoc/repos/planning-aware-trajectron/diffstack")
        sys.path.append("/home/yuxiaoc/repos/planning-aware-trajectron/diffstack/trajectron")
        sys.path.append("/home/yuxiaoc/repos/planning-aware-trajectron/diffstack/trajectron/trajectron")
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in diffstack_args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if diffstack_args.scene_freq_mult_eval else None

        # Offline Calculate Validation Scene Graphs
        if hyperparams['offline_scene_graph'] == 'yes':
            print(f"Rank {rank}: Offline calculating scene graphs")
            for i, scene in enumerate(tqdm(eval_scenes, desc='Validation Scenes', disable=(rank > 0))):
                scene.calculate_scene_graph(eval_env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])

        
        # Do cached indexing                                    
        if not os.path.exists(cached_eval_data_path):
            print (f"Rank {rank}: Create eval data cache: {cached_eval_data_path}")

            eval_env.nusc_maps = nusc_maps
            trajectron.set_environment(eval_env)

            eval_dataset = EnvironmentDataset(eval_env,
                                            hyperparams['state'],
                                            hyperparams['pred_state'],
                                            scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                            node_freq_mult=hyperparams['node_freq_mult_eval'],
                                            hyperparams=hyperparams,
                                            min_history_timesteps=hyperparams['minimum_history_length'],
                                            min_future_timesteps=hyperparams['prediction_horizon'],
                                            return_robot=not diffstack_args.incl_robot_node,
                                            num_workers=diffstack_args.indexing_workers,
                                            rank=rank)
            with torch.no_grad():                                            
                trajectron.augment_dataset_with_plan_info(eval_dataset, collate_fn=collate, batch_size=diffstack_args.batch_size, rank=rank, world_size=world_size)

            del eval_dataset.env.nusc_maps

            if rank == 0:
                with open(cached_eval_data_path, 'wb') as f:
                    dill.dump(eval_dataset, f, protocol=4) # For Python 3.6 and 3.8 compatability.
                print (f"Rank {rank}: cache eval files saved.")

            del eval_dataset
        torch.distributed.barrier()
        if rank == 0:
            print (f"All workers loading eval data. Use cache: {cached_eval_data_path}")
        
        with open(cached_eval_data_path, 'rb') as f:
            eval_env = dill.load(f).env
        
        self.eval_env = eval_env
        self.eval_scenes = eval_scenes
        attention_radius = dict()
        attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0
        attention_radius[(AgentType.PEDESTRIAN, AgentType.VEHICLE)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.PEDESTRIAN)] = 20.0
        attention_radius[(AgentType.VEHICLE, AgentType.VEHICLE)] = 30.0
        
        eval_dataset = UnifiedDataset(desired_data=["nusc_mini-mini_val"],
                                    history_sec=(hyperparams['maximum_history_length'] * 0.5, hyperparams['maximum_history_length'] * 0.5),
                                    future_sec=(hyperparams['prediction_horizon'], hyperparams['prediction_horizon']),
                                    standardize_data=False,
                                    agent_interaction_distances=attention_radius,
                                    incl_robot_future=hyperparams['incl_robot_node'],
                                    incl_map=False,
                                    only_types=[AgentType.VEHICLE],
                                    num_workers=diffstack_args.preprocess_workers,
                                    cache_location=os.path.join(diffstack_args.cache_dir, ".unified_data_cache"),
                                    data_dirs={"nusc_mini": raw_data_path}, #"/home/bivanovic/datasets/nuScenes"},
                                    verbose=True)
        self.eval_dataset = eval_dataset
        # eval_sampler = data.distributed.DistributedSampler(
        #     eval_dataset,
        #     num_replicas=dist.get_world_size(),
        #     rank=rank
        # )

        # eval_dataloader = data.DataLoader(eval_dataset,
        #                                     collate_fn=eval_dataset.get_collate_fn(),
        #                                     pin_memory=False if diffstack_args.device == 'cpu' else True,
        #                                     batch_size=diffstack_args.eval_batch_size,
        #                                     shuffle=False,
        #                                     num_workers=diffstack_args.preprocess_workers,
        #                                     sampler=eval_sampler,
        #                                     persistent_workers=diffstack_args.preprocess_workers > 0)


        # Pass on map reference for online preprocessing
        eval_env.nusc_maps = nusc_maps
        self.nusc_maps = nusc_maps
        trajectron.set_environment(eval_env)
        trajectron.set_annealing_params()
        self.trajectron = trajectron
        self.hyperparams = hyperparams
    def reformatting(self,new_form_batch):

        old_form_batch=reformat_batch(new_form_batch, self.eval_dataset, self.eval_env, self.hyperparams, self.cached_sample_scene_map)

        (first_history_index,
            x_t, y_t, x_st_t, y_st_t,
            neighbors_data_st,  # dict of lists. edge_type -> [batch][neighbor]: Tensor(time, statedim). Represetns 
            neighbors_edge_value,
            robot_traj_st_t,
            map, neighbors_future_data, plan_data) = old_form_batch
       
        scene_ids = plan_data['scene_ids']
        agent_states = list()
        for batch_i,scene_id in enumerate(scene_ids):
            
            # construct dummy observation from x_t and neighbors_data
            agent_states.append(dict())


            # local to global

            scene = self.find_scene(scene_id)
            x_t[batch_i,..., 0] += scene.x_min
            x_t[batch_i,..., 1] += scene.y_min
            if np.isnan(x_t).any():
                x_t = pad_nan(x_t)
            # this is due to the agent policy takes 10 steps history while the ego policy (diff stack) takes 8
            agent_states[batch_i][0] = x_t[batch_i,-9:]
            neighbor_x_t = new_form_batch.neigh_hist[batch_i][:new_form_batch.num_neigh[batch_i]]
            neighbor_x_t = torch.cat((neighbor_x_t,torch.zeros_like(neighbor_x_t[...,0:1])),-1)
            neighbor_x_t[...,0]+=scene.x_min
            neighbor_x_t[...,1]+=scene.y_min
            if np.isnan(neighbor_x_t).any():
                neighbor_x_t = pad_nan(neighbor_x_t.reshape(-1,neighbor_x_t.shape[-1])).reshape(*neighbor_x_t.shape)
            for i in range(neighbor_x_t.shape[0]):
                agent_states[batch_i][i+1] = neighbor_x_t[i,-9:]

            
            
        return agent_states, scene_ids
    def find_scene(self,scene_id):
        for scene in self.eval_scenes:
            if self.cached_sample_scene_map[scene.name] == scene_id:
                return scene
        assert False, f"not found scene {scene_id}"
    def eval(self):
        self.trajectron.eval()
    
    def get_action(self,obs,**kwargs)-> Tuple[Action, Dict]:
        state_histories, _ = self.reformatting(obs)
        scene_ids = obs.scene_ids
        max_hl = self.hyperparams['maximum_history_length']
        ph = self.hyperparams['prediction_horizon']
        past_and_future_trajlen = max_hl + ph            

        timesteps = np.arange(max_hl)
        state_fields_dict = {nk: {k: v for k, v in ndict.items() if k != "augment"} for nk, ndict in self.hyperparams['state'].items()}
        state_fields = state_fields_dict['VEHICLE']
        node_type = self.eval_env.NodeType.VEHICLE
        edge_types = [edge_type for edge_type in self.eval_env.get_edge_types() if edge_type[0] is node_type]   
        
        batch_pos = list()
        batch_yaw = list()
        for batch_i,scene_id in enumerate(scene_ids):

            scene = self.find_scene(scene_id)

            # Create nodes
            node_dict = {}

            
            for agent_id, statehist in state_histories[batch_i].items():
                # const velocity as future
                # state: x, y, yaw, vel; control: yaw_rate, acc
                x0 = torch.tensor([[statehist[-1, 0], statehist[-1, 1], statehist[-1, -2], torch.linalg.norm(statehist[-1, 2:4])]])  # (b, 4)
                u_zeros = torch.zeros((ph+1, x0.shape[0], 2), dtype=x0.dtype, device=x0.device)
                
                x_constvel = mpc_util.get_traj(ph+1, u_zeros, x0, self.trajectron.dyn_obj)  # (T+1, b, 4)
                pred_constvel = x_constvel[1:]  # (T, b, 2)
                pred_constvel = pred_constvel.transpose(0, 1).squeeze(0)  # (T, 2) if b = 1
                
                vx = pred_constvel[..., -1] * torch.cos(pred_constvel[..., -2])
                vy = pred_constvel[..., -1] * torch.sin(pred_constvel[..., -2])
                
                ax = torch.diff(vx, prepend=vx[[0]] - (vx[[1]] - vx[[0]])) / 0.5
                ay = torch.diff(vy, prepend=vy[[0]] - (vy[[1]] - vy[[0]])) / 0.5
                
                constvel_future = torch.stack([pred_constvel[:, 0], pred_constvel[:, 1], vx, vy, ax, ay, pred_constvel[:, -2], torch.zeros_like(vx)], dim=1)

                stathist_and_future = torch.cat((statehist, constvel_future), dim=0)
                x_global, y_global, vx, vy, ax, ay, yaw, _ = stathist_and_future.transpose(1, 0)  # (t, 8) --> (8, t)
                # print (x)
                node = create_node(x_global, y_global, vx, vy, ax, ay, yaw, self.nusc_maps[scene.map_name], scene, node_type=self.eval_env.NodeType.VEHICLE, node_id=str(agent_id), is_robot=(agent_id==0))
                # print (node.data[0:1, (('position', 'x'))])
                node_dict[agent_id] = node

            # replace nodes in the scene
            scene.nodes = list(node_dict.values())
            # get rid of the old scene graph
            scene.temporal_scene_graph = None
            
            # Choose ego, prediction, and other agents
            all_states = np.stack([v for k, v in state_histories[batch_i].items()], axis=0)  # N, T, 8 
            # Turn into local coordinates]
            all_states[..., 0] -= scene.x_min
            all_states[..., 1] -= scene.y_min
            ego = all_states[0]

            # Find closest agent to do prediction for
            dist = np.linalg.norm(ego[None, -1, :2] - all_states[1:, -1, :2], axis=-1)
            closest_i = np.argmin(dist)
            closest_agent_id = list(node_dict.keys())[closest_i+1]  # +1 because we skipped ego
            pred_node = node_dict[closest_agent_id]

            # Create dummy training batch
            with torch.no_grad():
            # TODO make sure history and future are aligned, i.e.  we have the current state as history (and repeated twice as t+1, t+2)
            
                sample = get_node_timestep_data(self.eval_env, scene, max_hl, pred_node, state_fields_dict,
                                self.hyperparams['pred_state'], edge_types, 
                                max_hl, ph, self.hyperparams, self.nusc_maps)
                # replace planning agent
                sample = self.trajectron.augment_sample_with_dummy_plan_info(sample, ego_traj=ego)

                batch = collate([sample])

                eval_loss_node_type, plot_data = self.trajectron.predict_and_evaluate_batch(batch, node_type, max_hl, return_plot_data=True)


            # TODO need to handle invalid planning case, e.g. give up episode or take random action

            plan_metric_dict, plan_info_dict = plot_data['plan']
    
            plan_x = plan_info_dict['plan_x'].squeeze(1)  # x, y, yaw, vel
            plan_u = plan_info_dict['plan_u'].squeeze(1)  # dyaw, acc
            plan_valid = eval_loss_node_type['plan_valid'].squeeze(0) and eval_loss_node_type['plan_converged'].squeeze(0)

            # local to global
            plan_x = TensorUtils.to_numpy(plan_x[1:])
            curr_yaw = ego[-1,6]
            curr_pos = ego[-1,:2]
            agent_from_world = np.array(
                    [
                        [np.cos(curr_yaw), np.sin(curr_yaw)],
                        [-np.sin(curr_yaw), np.cos(curr_yaw)],
                    ]
                )

            rel_xy = plan_x[:,:2]-curr_pos
            local_xy = (agent_from_world@(rel_xy.T)).T
            local_yaw = plan_x[:,2]-curr_yaw
            batch_pos.append(local_xy)
            batch_yaw.append(local_yaw[:,np.newaxis])
        print(local_xy)
        action = Action(positions=TensorUtils.to_torch(np.stack(batch_pos,0),device=self.device),
                        yaws=TensorUtils.to_torch(np.stack(batch_yaw,0),device=self.device))
        action_info = dict()
        return action, action_info

def pad_nan(x):
    if isinstance(x,np.ndarray):
        flag = ~(np.isnan(x)).any(axis=-1)
        if flag.sum()==0:
            x = np.zeros_like(x)
        else:
            idx = np.where(flag)[0]
            nan_idx = np.where(~flag)[0]
            reps = [nan_idx.shape[0]]+[1]*(x.ndim-1)
            x[nan_idx] = np.tile(x[idx[0]],reps)
    elif isinstance(x,torch.Tensor):
        flag = ~(torch.isnan(x)).any(dim=-1)
        if flag.sum()==0:
            x = torch.zeros_like(x)
        else:
            idx = torch.where(flag)[0]
            nan_idx = torch.where(~flag)[0]
            reps = [nan_idx.shape[0]]+[1]*(x.ndim-1)
            x[nan_idx] = torch.tile(x[idx[0]],reps)
    return x



def create_node(global_x, global_y, vx, vy, ax, ay, yaw, nusc_map, scene, node_type, node_id="custom", is_robot=False, ):
    # assumes global x, y
    v = np.stack((vx, vy), axis=-1)
    v_norm = np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1, keepdims=True)
    heading_v = np.divide(v, v_norm, out=np.zeros_like(v), where=(v_norm > 1.))
    heading_x = heading_v[:, 0]
    heading_y = heading_v[:, 1]

    # Use heading from relative motion for moving vehicles (at least 1m/s)
    heading_from_v = np.arctan2(vy, vx)
    heading = np.where(v_norm.squeeze(1) > 1.0, heading_from_v, yaw)

    a_norm = np.divide(ax*vx + ay*vy, v_norm[..., 0], out=np.zeros_like(ax), where=(v_norm[..., 0] > 1.))
    a_control = np.linalg.norm(np.stack((ax, ay), axis=-1), axis=-1)
    d_heading = derivative_of(heading, scene.dt, radian=True)

    # Plan features
    offset_xyh = np.array([scene.x_min, scene.y_min, 0.])
    global_xyh = np.stack((global_x, global_y, heading), axis=-1)
    if np.isnan(global_xyh).any():
        global_xyh=pad_nan(global_xyh)
    xyh = global_xyh - offset_xyh[None]
    x = xyh[..., 0]
    y = xyh[..., 1]
    # xyh = np.stack((x, y, heading), axis=-1)
    # global_xyh = xyh + offset_xyh[None]

    

    lane_ref_points = get_lane_reference_points(global_xyh, nusc_map)

    lane_ref_points -= offset_xyh[None, None]

    # Lane simplified using precomputed lane reference points (already local)
    lane_t_xyh2 = []
    for t in range(global_x.shape[0]):
        if lane_ref_points.shape[1] == 0:
            lane_xyh = np.full((3, ), fill_value=np.nan)
        else:
            lane_xyh = pred_metric_utils.lane_frenet_features_simple(xyh[t], lane_ref_points[t, :1])                        
        lane_t_xyh2.append(lane_xyh)
    lane_t_xyh2 = np.stack(lane_t_xyh2)  # (T, 3)
    if np.isnan(lane_t_xyh2).any():
        import pdb
        pdb.set_trace()
    lane_t_xyh = lane_t_xyh2
    projected_t_xyh = lane_t_xyh

    fitted_controls_traj = np.full((global_x.shape[0], CONTROL_FIT_H, 2), fill_value=np.nan)

    data_dict = {('position', 'x'): x,
                    ('position', 'y'): y,
                    ('velocity', 'x'): vx,
                    ('velocity', 'y'): vy,
                    ('velocity', 'norm'): np.linalg.norm(np.stack((vx, vy), axis=-1), axis=-1),
                    ('acceleration', 'x'): ax,
                    ('acceleration', 'y'): ay,
                    ('acceleration', 'norm'): a_norm,
                    ('acceleration', 'norm2'): a_control,
                    ('heading', 'x'): heading_x,
                    ('heading', 'y'): heading_y,
                    ('heading', '°'): heading,
                    ('heading', 'd°'): d_heading,
                    ('lane', 'x'): lane_t_xyh[:, 0],
                    ('lane', 'y'): lane_t_xyh[:, 1],
                    ('lane', '°'): lane_t_xyh[:, 2],
                    ('projected', 'x'): projected_t_xyh[:, 0],
                    ('projected', 'y'): projected_t_xyh[:, 1],
                    ('projected', '°'): projected_t_xyh[:, 2],
                    }
    for t in range(CONTROL_FIT_H):
        data_dict[('control_traj_dh', "t"+str(t))] = fitted_controls_traj[:, t, 0]
        data_dict[('control_traj_a', "t"+str(t))] = fitted_controls_traj[:, t, 1]

    node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

    node = Node(node_type=node_type, node_id=node_id, data=node_data,
                frequency_multiplier=1., extra_data={'lane_points': lane_ref_points})
    node.first_timestep = np.ones((), dtype=np.int32) * 0
    node.is_robot = is_robot
    return node


def main():
    
    diffstack_args = dict(augment=False, batch_size=256, bias_predictions=False, 
                      cache_dir='/home/yuxiaoc/data/cache', 
                      conf='/home/yuxiaoc/repos/planning-aware-trajectron/diffstack/trajectron/config/plan6_ego_nofilt.json', 
                      data_dir='/home/yuxiaoc/data/nuscenes_mini_plantpp_v5', dataset_version='', 
                      debug=False, device='cuda:0', dynamic_edges='yes', edge_addition_filter=[0.25, 0.5, 0.75, 1.0], 
                      edge_influence_combine_method='attention', edge_removal_filter=[1.0, 0.0], 
                      edge_state_combine_method='sum', eval_batch_size=256, eval_data_dict='nuScenes_mini_val.pkl', 
                      eval_every=1, experiment='diffstack-def', incl_robot_node=False, indexing_workers=0, 
                      interactive=False, k_eval=25, learning_rate=None, load='', load2='', local_rank=0, 
                      log_dir='../experiments/logs', log_tag='', lr_step=None, map_encoding=False, 
                      no_edge_encoding=False, no_plan_train=False, no_train_pred=False, node_freq_mult_eval=False, 
                      node_freq_mult_train=False, offline_scene_graph='yes', override_attention_radius=[], 
                      plan_cost='', plan_cost_for_gt='', plan_dt=0.0, plan_init='fitted', plan_loss='mse', 
                      plan_loss_scale_end=-1, plan_loss_scale_start=-1, plan_loss_scaler=10000.0, 
                      plan_lqr_eps=0.01, planner='', pred_loss_scaler=1.0, pred_loss_temp=1.0, 
                      pred_loss_weights='none', preprocess_workers=0, profile='', runmode='train', 
                      save_every=1, scene_freq_mult_eval=False, scene_freq_mult_train=False, 
                      scene_freq_mult_viz=False, seed=123, train_data_dict='train.pkl', train_epochs=1, 
                      train_plan_cost=False, vis_every=0)
    policy = DiffStackPolicy(diffstack_args,device="cuda")

if __name__ == '__main__':
    main()
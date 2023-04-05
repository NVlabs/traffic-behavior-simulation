from dataclasses import asdict
from collections import defaultdict
from posixpath import split
import torch
import numpy as np
from copy import deepcopy
from typing import List
from trajdata import UnifiedDataset, AgentBatch, AgentType
from trajdata.simulation import SimulationScene
from trajdata.simulation import sim_metrics
from scripts.parse_results import parse

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_trajdata
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.policies.common import RolloutAction, Action
from tbsim.utils.geometry_utils import transform_points_tensor
from tbsim.utils.timer import Timers
from tbsim.utils.trajdata_utils import parse_trajdata_batch, get_drivable_region_map
from tbsim.utils.rollout_logger import RolloutLogger
from torch.nn.utils.rnn import pad_sequence
from trajdata.data_structures.state import StateArray

agent_types=[AgentType.UNKNOWN,AgentType.VEHICLE,AgentType.PEDESTRIAN,AgentType.BICYCLE,AgentType.MOTORCYCLE]

class EnvUnifiedSimulation(BaseEnv, BatchedEnv):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            renderer=None,
            restart_track_id=True,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._log_data = log_data
        self.logger = None
        self.restart_track_id = restart_track_id

    def update_random_seed(self, seed):
        self._npr = np.random.RandomState(seed=seed)
    



    @property
    def current_scene_names(self):
        return deepcopy([scene.scene.name for scene in self._current_scenes])

    @property
    def current_num_agents(self):
        return sum(len(scene.agents) for scene in self._current_scenes)

    def reset_multi_episodes_metrics(self):
        for v in self._metrics.values():
            v.multi_episode_reset()

    @property
    def current_agent_scene_index(self):
        si = []
        for scene_i, scene in zip(self.current_scene_index, self._current_scenes):
            si.extend([scene_i] * len(scene.agents))
        return np.array(si, dtype=np.int64)

    @property
    def current_agent_track_id(self):
        if self.restart_track_id:
            id_by_scene=defaultdict(lambda :0)
            track_id = np.zeros_like(self.current_agent_scene_index)
            for i in range(self.current_agent_scene_index.shape[0]):
                track_id[i] = id_by_scene[self.current_agent_scene_index[i]]
                id_by_scene[self.current_agent_scene_index[i]]+=1
            return track_id
        else:
            return np.arange(self.current_num_agents)

    @property
    def current_scene_index(self):
        return self._current_scene_indices.copy()

    @property
    def current_agent_names(self):
        names = []
        for scene in self._current_scenes:
            names.extend([a.name for a in scene.agents])
        return names

    @property
    def num_instances(self):
        return self._num_scenes

    @property
    def total_num_scenes(self):
        return self._num_total_scenes

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    @property
    def horizon(self):
        return self._env_config.simulation.num_simulation_steps
    
    def get_gt_action(self, obs):
        ego = Action(
            positions=obs["ego"]["target_positions"],
            yaws=obs["ego"]["target_yaws"]
        ) if "ego" in obs else None
        agents = Action(
            positions=obs["agents"]["target_positions"],
            yaws=obs["agents"]["target_yaws"]
        ) if "agents" in obs else None
        return RolloutAction(ego=ego, agents=agents)

    def _disable_offroad_agents(self, scene):
        obs = scene.get_obs()
        obs = parse_trajdata_batch(obs)
        drivable_region = get_drivable_region_map(obs["maps"])
        raster_pos = transform_points_tensor(obs["centroid"][:, None], obs["raster_from_world"])[:, 0]
        valid_agents = []
        for i, rpos in enumerate(raster_pos):
            if scene.agents[i].name == "ego" or drivable_region[i, int(rpos[1]), int(rpos[0])].item() > 0:
                valid_agents.append(scene.agents[i])

        scene.agents = valid_agents
    
    def add_new_agents(self,agent_data_by_scene):
        for sim_scene,agent_data in agent_data_by_scene.items():
            if sim_scene not in self._current_scenes:
                continue
            if len(agent_data)>0:
                sim_scene.add_new_agents(agent_data)

    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        if scene_indices is None:
            # randomly sample a batch of scenes for close-loop rollouts
            all_indices = np.arange(self._num_total_scenes)
            scene_indices = self._npr.choice(
                all_indices, size=(self.num_instances,), replace=False
            )

        scene_info = [self.dataset.get_scene(i) for i in scene_indices]

        self._num_scenes = len(scene_info)
        self._current_scene_indices = scene_indices

        assert (
                np.max(scene_indices) < self._num_total_scenes
                and np.min(scene_indices) >= 0
        )
        if start_frame_index is None:
            start_frame_index = self._env_config.simulation.start_frame_index
        self._current_scenes = []
        for i, si in enumerate(scene_info):
            sim_scene: SimulationScene = SimulationScene(
                env_name=self._env_config.name,
                scene_name=si.name,
                scene=si,
                dataset=self.dataset,
                init_timestep=start_frame_index,
                freeze_agents=True,
                return_dict=True,
                # vectorize_lane=self.dataset.vectorize_lane,
            )
            sim_scene.reset()
            self._disable_offroad_agents(sim_scene)
            self._current_scenes.append(sim_scene)

        self._frame_index = 0
        self._cached_observation = None
        self._cached_raw_observation = None
        self._done = False

        obs_keys_to_log = [
            "centroid",
            "yaw",
            "curr_speed",
            "extent",
            "world_from_agent",
            "scene_index",
            "track_id"
        ]
        self.logger = RolloutLogger(obs_keys=obs_keys_to_log)

        for v in self._metrics.values():
            v.reset()

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = [i for i, name in enumerate(self.current_agent_names) if name == "ego"]
        for i in ego_inds:
            im = render_state_trajdata(
                batch=self.get_observation()["agents"],
                batch_idx=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        agents = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )

        return RolloutAction(agents=agents)

    def get_info(self):
        map_info = {scene.scene_name: f"{scene.cache.scene.env_name}:{scene.cache.scene.location}" for scene in self._current_scenes}
        info = dict(scene_index=self.current_scene_names,map_info=map_info)
        if self._log_data:
            sim_buffer = self.logger.get_serialized_scene_buffer()
            sim_buffer = [sim_buffer[k] for k in self.current_scene_index]
            info["buffer"] = sim_buffer
            # self.logger.get_trajectory()
        return info

    def get_multi_episode_metrics(self):
        metrics = dict()
        for met_name, met in self._metrics.items():
            met_vals = met.get_multi_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            elif met_vals is not None:
                metrics[met_name] = met_vals

        return TensorUtils.detach(metrics)

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """
        metrics = dict()
        # get ADE and FDE from SimulationScene
        metrics["ade"] = np.zeros(self.num_instances)
        metrics["fde"] = np.zeros(self.num_instances)
        for i, scene in enumerate(self._current_scenes):
            mets_per_agent = scene.get_metrics([sim_metrics.ADE(), sim_metrics.FDE()])
            metrics["ade"][i] = np.array(list(mets_per_agent["ade"].values())).mean()
            metrics["fde"][i] = np.array(list(mets_per_agent["fde"].values())).mean()

        # aggregate per-step metrics
        for met_name, met in self._metrics.items():
            met_vals = met.get_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            else:
                metrics[met_name] = met_vals

        for k in metrics:
            assert metrics[k].shape == (self.num_instances,)
        return TensorUtils.detach(metrics)

    def get_observation_by_scene(self):
        obs = self.get_observation()["agents"]
        obs_by_scene = []
        obs_scene_index = self.current_agent_scene_index
        for i in range(self.num_instances):
            obs_by_scene.append(TensorUtils.map_ndarray(obs, lambda x: x[obs_scene_index == i]))
        return obs_by_scene
        
    def get_observation(self):
        if self._cached_observation is not None:
            return self._cached_observation

        self.timers.tic("get_obs")

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        agent_obs["env_name"] = [self._current_scenes[self.current_scene_index.index(i)].env_name for i in agent_obs["scene_index"]]
        

        # cache observations
        self._cached_observation = dict(agents=agent_obs)
        self.timers.toc("get_obs")

        return self._cached_observation


    def get_observation_skimp(self):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=False))
        agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        agent_obs = parse_trajdata_batch(agent_obs)
        agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
        agent_obs["scene_index"] = self.current_agent_scene_index
        agent_obs["track_id"] = self.current_agent_track_id
        self.timers.toc("obs_skimp")
        return dict(agents=agent_obs)

    def _add_per_step_metrics(self, obs):
        for k, v in self._metrics.items():
            v.add_step(obs, self.current_scene_index)

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")

        obs = self.get_observation()["agents"]   
        # record metrics
        # self._add_per_step_metrics(obs)

        action = step_actions.agents.to_dict()
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        self._add_per_step_metrics(obs)
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp()
            
            if self._log_data:
                action_to_log = RolloutAction(
                    agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                    agents_info=step_actions.agents_info
                )
                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["yaw"][idx]
                    curr_pos = obs["centroid"][idx]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[2] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        print("invalid action!")
                    scene_action[agent.name] = next_state
                    idx += 1
                StateArray_action = dict()
                for k,v in scene_action.items():
                    xyzh = np.insert(v,2,0.0)
                    StateArray_action[k] = StateArray.from_array(xyzh,"x,y,z,h")
                    
                scene.step(StateArray_action, return_obs=False)

        self._cached_observation = None

        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
        print(self.timers)

    def step(self, actions: RolloutAction, num_steps_to_take: int = 1, render=False):
        """
        Step the simulation with control inputs

        Args:
            actions (RolloutAction): action for controlling ego and/or agents
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
            render (bool): whether to render state and actions and return renderings
        """
        actions = actions.to_numpy()
        renderings = []
        if render:
            renderings.append(self.render(actions))
        self._step(step_actions=actions, num_steps_to_take=num_steps_to_take)
        return renderings
    
    def adjust_scene(self,adjust_plan):
        agent_data_by_scene = dict()
        for simscene in self._current_scenes:
            if simscene.scene.name in adjust_plan:
                adjust_plan_i = adjust_plan[simscene.scene.name]
                if adjust_plan_i["remove_existing_neighbors"]["flag"] and not adjust_plan_i["remove_existing_neighbors"]["executed"]:
                    simscene.agents = [agent for agent in simscene.agents if agent.name=="ego"]
                    adjust_plan_i["remove_existing_neighbors"]["executed"]=True
                agent_data=list()
                for agent_data_i in adjust_plan_i["agents"]:
                    if not agent_data_i["executed"]:
                        agent_data.append([agent_data_i["name"],
                                        np.array(agent_data_i["agent_state"]),
                                        agent_data_i["initial_timestep"],
                                        agent_types[agent_data_i["agent_type"]],
                                        np.array(agent_data_i["extent"]),
                                        ])
                        agent_data_i["executed"]=True
                agent_data_by_scene[simscene] = agent_data

        self.add_new_agents(agent_data_by_scene)
    

        
class EnvSplitUnifiedSimulation(EnvUnifiedSimulation):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset: UnifiedDataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            log_data=True,
            split_ego = False,
            renderer=None,
            restart_track_id=True,
            parse_obs = True,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with UnifiedDataset, with the capability of spliting ego and agent observations

        Args:
            env_config (NuscEnvConfig): a Config object specifying the behavior of the simulator
            num_scenes (int): number of scenes to run in parallel
            dataset (UnifiedDataset): a UnifiedDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
            split_ego (bool): if set to True, split ego out as the ego observation
            parse_obs (bool or dict): whether to parse the ego and agent observation or not
        """
        print(env_config)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset
        self._env_config = env_config

        self._num_total_scenes = dataset.num_scenes()
        self._num_scenes = num_scenes
        self.split_ego = split_ego
        self.parse_obs = parse_obs

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scenes: List[SimulationScene] = None # corresponding dataset of the scenes
        self._current_scene_indices = None

        self._frame_index = 0
        self._done = False
        self._prediction_only = prediction_only

        self._cached_observation = None
        self._cached_raw_observation = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._log_data = log_data
        self.restart_track_id = restart_track_id
        self.logger = None


    def reset(self, scene_indices: List = None, start_frame_index = None):
        """
        Reset the previous simulation episode. Randomly sample a batch of new scenes unless specified in @scene_indices

        Args:
            scene_indices (List): Optional, a list of scene indices to initialize the simulation episode
        """
        super(EnvSplitUnifiedSimulation,self).reset(scene_indices,start_frame_index)
        self._cached_raw_observation = None

    def render(self, actions_to_take):
        scene_ims = []
        ego_inds = [i for i, name in enumerate(self.current_agent_names) if name == "ego"]
        for i in ego_inds:
            im = render_state_trajdata(
                batch=self.get_observation(split_ego=False)["agents"],
                batch_idx=i,
                action=actions_to_take
            )
            scene_ims.append(im)
        return np.stack(scene_ims)

    def get_random_action(self):
        ac = self._npr.randn(self.current_num_agents, 1, 3)
        if self.split_ego:
            ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
            ego_action = Action(
                positions=ac[ego_idx, :, :2],
                yaws=ac[ego_idx, :, 2:3]
            )
            agent_action = Action(
                positions=ac[agent_idx, :, :2],
                yaws=ac[agent_idx, :, 2:3]
            )
            return RolloutAction(ego=ego_action,agents=agent_action)
        else:
            agents = Action(
                positions=ac[:, :, :2],
                yaws=ac[:, :, 2:3]
            )

            return RolloutAction(agents=agents)


    def get_observation(self,split_ego=None,return_raw=False):
        if split_ego is None:
            split_ego = self.split_ego
        if return_raw:
            if self._cached_raw_observation is not None:
                return self._cached_raw_observation
        else:
            if self._cached_observation is not None:
                if split_ego and "ego" in self._cached_observation:
                    return self._cached_observation
                elif not split_ego and "ego" not in self._cached_observation:
                    return self._cached_observation
                else:
                    self._cached_observation = None
                    self._cached_raw_observation = None

        self.timers.tic("get_obs")

        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False))
        self._cached_raw_observation = raw_obs
        if return_raw:
            return raw_obs
        if split_ego:
            # obtain index of ego and agents    
            ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
            agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
            # raw_obs is the raw trajdata batch_element object without collation
            ego_obs_raw = [raw_obs[idx] for idx in ego_idx]
            # call the collate function to turn batch_element into trajdata batch object
            ego_obs_collated = self.dataset.get_collate_fn(return_dict=True)(ego_obs_raw)
            agent_obs_raw = [raw_obs[idx] for idx in agent_idx]
            # call the collate function to turn batch_element into trajdata batch object
            if len(agent_obs_raw)>0:
                agent_obs_collated = self.dataset.get_collate_fn(return_dict=True)(agent_obs_raw)
            else:
                agent_obs_collated = dict()
                for k,v in ego_obs_collated.items():
                    if isinstance(v,torch.Tensor):
                        if v.dim()>=1:
                            agent_obs_collated[k] = torch.zeros([0,*v.shape[1:]],dtype=v.dtype,device=v.device)
                        else:
                            agent_obs_collated[k] = torch.zeros(0,device=v.device)
                    elif isinstance(v,list):
                        agent_obs_collated[k] = []
                    else:
                        agent_obs_collated[k] = None
            
            # parse_obs can be True (parse both ego and agent), or False (parse neither), or dictionary that determines whether to parse ego or agent observation
            if self.parse_obs==True:
                parse_plan = dict(ego=True,agent=True)
            elif self.parse_obs==False:
                parse_plan = dict(ego=False,agent=False)
            elif isinstance(self.parse_obs,dict):
                parse_plan = self.parse_obs
            if parse_plan["ego"]:
                ego_obs = parse_trajdata_batch(ego_obs_collated)
                ego_obs = TensorUtils.to_numpy(ego_obs,ignore_if_unspecified=True)
                ego_obs["scene_index"] = self.current_agent_scene_index[ego_idx]
                ego_obs["track_id"] = self.current_agent_track_id[ego_idx]
                # ego_obs["env_name"] = [self._current_scenes[i].env_name for i in ego_obs["scene_index"]]
            else:
                # put collated observation into AgentBatch object from trajdata
                ego_obs = AgentBatch(**ego_obs_collated)
            if parse_plan["agent"]:
                agent_obs = parse_trajdata_batch(agent_obs_collated)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                if len(agent_idx)>0:
                    agent_obs["scene_index"] = self.current_agent_scene_index[agent_idx]
                    agent_obs["track_id"] = self.current_agent_track_id[agent_idx]
                else:
                    agent_obs["scene_index"] = []
                    agent_obs["track_id"] = []
                # agent_obs["env_name"] = [self._current_scenes[i].env_name for i in agent_obs["scene_index"]]
            else:
                # put collated observation into AgentBatch object from trajdata
                agent_obs = AgentBatch(**agent_obs_collated)
            self._cached_observation = dict(ego=ego_obs,agents=agent_obs)
        else:
            # if ego is not splitted out, then either parse all observa tion or do not parse any observation.
            assert isinstance(self.parse_obs,bool)
            agent_obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
            if self.parse_obs:
                agent_obs = parse_trajdata_batch(agent_obs)
                agent_obs = TensorUtils.to_numpy(agent_obs,ignore_if_unspecified=True)
                agent_obs["scene_index"] = self.current_agent_scene_index
                agent_obs["track_id"] = self.current_agent_track_id
                # agent_obs["env_name"] = [self._current_scenes[i].env_name for i in agent_obs["scene_index"]]
            else:
                agent_obs = AgentBatch(**agent_obs)
            self._cached_observation = dict(agents=agent_obs)

        self.timers.toc("get_obs")
        return self._cached_observation

    
    def combine_action(self,step_actions):
        # combine ego and agent actions
        if step_actions.agents is None:
            return RolloutAction(agents=step_actions.ego,agents_info=step_actions.ego_info)
        ego_action = step_actions.ego.to_dict()
        agent_action = step_actions.agents.to_dict()
        ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
        min_length = min(ego_action["positions"].shape[1],agent_action["positions"].shape[1])
        combined_positions = np.zeros([len(self.current_agent_names),min_length,2])
        combined_yaws = np.zeros([len(self.current_agent_names),min_length,1])
        combined_positions[ego_idx] = ego_action["positions"][:,:min_length]
        combined_positions[agent_idx] = agent_action["positions"][:,:min_length]
        combined_yaws[ego_idx] = ego_action["yaws"][:,:min_length]
        combined_yaws[agent_idx] = agent_action["yaws"][:,:min_length]
        return RolloutAction(agents=Action(positions=combined_positions,yaws=combined_yaws),agents_info=step_actions.agents_info)


    def combine_obs(self,ego_obs,agent_obs):
        # combining ego and agent observation, not really used.
        ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
        agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
        bs = len(self.current_agent_names)
        combined_obs = dict()
        for k,v in ego_obs.items():
            if k in agent_obs and v is not None:
                combined_v = np.zeros([bs,*v.shape[1:]])
                combined_v[ego_idx]=ego_obs[k]
                combined_v[agent_idx]=agent_obs[k]
                combined_obs[k]=combined_v
        return combined_obs
    def get_observation_skimp(self,split_ego=True):
        self.timers.tic("obs_skimp")
        raw_obs = []
        for si, scene in enumerate(self._current_scenes):
            raw_obs.extend(scene.get_obs(collate=False, get_map=False))
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        self.timers.toc("obs_skimp")
        keys_to_take = ['agent_type', 'image', 'drivable_map', 'target_positions',\
             'target_yaws', 'target_availabilities', 'history_positions', 'history_yaws', 'history_availabilities',\
             'curr_speed', 'centroid', 'yaw', 'type', 'extent', 'raster_from_agent', 'agent_from_raster',\
             'raster_from_world', 'agent_from_world', 'world_from_agent', 'scene_index', 'track_id']
        obs_selected = {k:v for k,v in obs.items() if k in keys_to_take}
        if split_ego:
            ego_mask = [name=="ego" for name in self.current_agent_names]
            agents_mask = [name!="ego" for name in self.current_agent_names]
            ego_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[ego_mask])
            agents_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[agents_mask])
            
            return dict(ego=ego_obs,agents=agents_obs)
        else:
            return dict(agents=obs_selected)
    def _add_per_step_metrics(self, obs):

        ego_mask = [name=="ego" for name in self.current_agent_names]
        agents_mask = [name!="ego" for name in self.current_agent_names]
        keys_to_take = ['agent_type', 'image', 'drivable_map', 'target_positions',\
             'target_yaws', 'target_availabilities', 'history_positions', 'history_yaws', 'history_availabilities',\
             'curr_speed', 'centroid', 'yaw', 'type', 'extent', 'raster_from_agent', 'agent_from_raster',\
             'raster_from_world', 'agent_from_world', 'world_from_agent', 'scene_index', 'track_id']
        obs_selected = {k:v for k,v in obs.items() if k in keys_to_take}
        ego_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[ego_mask])
        agents_obs = TensorUtils.map_ndarray(obs_selected, lambda x: x[agents_mask])
        ego_map_names = [obs["map_names"][i] for i,name in enumerate(self.current_agent_names) if name=="ego"]
        agents_map_names = [obs["map_names"][i] for i,name in enumerate(self.current_agent_names) if name!="ego"]
        ego_obs["map_names"] = ego_map_names
        agents_obs["map_names"] = agents_map_names
        for k, v in self._metrics.items():
            if k.startswith("ego"):
                v.add_step(ego_obs, self._current_scene_indices)
            elif k.startswith("agents"):
                v.add_step(agents_obs, self._current_scene_indices)
            elif k.startswith("all"):
                v.add_step(obs, self._current_scene_indices)
            else:
                raise KeyError("Invalid metrics name {}".format(k))

    def _step(self, step_actions: RolloutAction, num_steps_to_take):
        if self.is_done():
            raise SimulationException("Cannot step in a finished episode")
        self.timers.tic("_step")
        # to bypass all the ego split, collation and parsing, directly get raw obs
        raw_obs = self.get_observation(split_ego=False,return_raw=True)
        obs = self.dataset.get_collate_fn(return_dict=True)(raw_obs)
        # always parse when stepping
        obs = parse_trajdata_batch(obs)
        obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
        obs["scene_index"] = self.current_agent_scene_index
        obs["track_id"] = self.current_agent_track_id
        # obs = {k:v for k,v in obs.items() if not isinstance(v,list)}

        # record metrics
        self._add_per_step_metrics(obs)
        if step_actions.has_ego:
            combined_step_actions = self.combine_action(step_actions)
            action = combined_step_actions.agents.to_dict()
        else:
            action = step_actions.agents.to_dict()
        
        assert action["positions"].shape[0] == obs["centroid"].shape[0]
        for action_index in range(num_steps_to_take):
            if action_index >= action["positions"].shape[1]:  # GT actions may be shorter
                self._done = True
                self._frame_index += action_index
                self._cached_observation = None
                self._cached_raw_observation = None
                return
            # # log state and action
            obs_skimp = self.get_observation_skimp(split_ego=True)
            # self._add_per_step_metrics(obs_skimp["agents"])
            if self._log_data:
                if step_actions.has_ego:
                    ego_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name=="ego"])
                    agent_idx = np.array([i for i,name in enumerate(self.current_agent_names) if name!="ego"])
                    action_t = TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(dict(positions=action_t["positions"][agent_idx],yaws=action_t["yaws"][agent_idx])) if len(agent_idx)>0 else None,
                        agents_info=step_actions.agents_info if len(agent_idx)>0 else None,
                        ego = Action.from_dict(dict(positions=action_t["positions"][ego_idx],yaws=action_t["yaws"][ego_idx])),
                        ego_info=step_actions.ego_info,
                    )
                else:
                    action_to_log = RolloutAction(
                        agents=Action.from_dict(TensorUtils.map_ndarray(action, lambda x: x[:, action_index:])),
                        agents_info=step_actions.agents_info
                    )

                self.logger.log_step(obs_skimp, action_to_log)

            idx = 0
            for scene in self._current_scenes:
                scene_action = dict()
                for agent in scene.agents:
                    curr_yaw = obs["yaw"][idx]
                    curr_pos = obs["centroid"][idx]
                    world_from_agent = np.array(
                        [
                            [np.cos(curr_yaw), np.sin(curr_yaw)],
                            [-np.sin(curr_yaw), np.cos(curr_yaw)],
                        ]
                    )
                    next_state = np.zeros(3, dtype=obs["agent_fut"].dtype)
                    if not np.any(np.isnan(action["positions"][idx, action_index])):  # ground truth action may be NaN
                        next_state[:2] = action["positions"][idx, action_index] @ world_from_agent + curr_pos
                        next_state[2] = curr_yaw + action["yaws"][idx, action_index, 0]
                    else:
                        print("invalid action!")
                    scene_action[agent.name] = next_state
                    idx += 1
                StateArray_action = dict()
                for k,v in scene_action.items():
                    xyzh = np.insert(v,2,0.0)
                    StateArray_action[k] = StateArray.from_array(xyzh,"x,y,z,h")
                scene.step(StateArray_action, return_obs=False)

        self._cached_observation = None
        self._cached_raw_observation = None
        self.timers.toc("_step")
        if self._frame_index + num_steps_to_take >= self.horizon:
            self._done = True
        else:
            self._frame_index += num_steps_to_take
        print(self.timers)
        


import numpy as np
from copy import deepcopy
from typing import List, Dict
from torch.utils.data.dataloader import default_collate
from collections import OrderedDict

from l5kit.simulation.unroll import ClosedLoopSimulator, SimulationOutput
from l5kit.cle.metrics import DisplacementErrorL2Metric
from l5kit.data import filter_agents_by_frames

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.vis_utils import render_state_l5kit_agents_view, render_state_l5kit_ego_view
from tbsim.envs.base import BaseEnv, BatchedEnv, SimulationException
from tbsim.l5kit.simulation_dataset import SimulationDataset, SimulationConfig
from tbsim.policies.common import RolloutAction, Action
from tbsim.utils.timer import Timers
from tbsim.utils.rollout_logger import RolloutLogger


class EnvL5KitSimulation(BaseEnv, BatchedEnv):
    def __init__(
            self,
            env_config,
            num_scenes,
            dataset,
            seed=0,
            prediction_only=False,
            metrics=None,
            skimp_rollout=False,
            renderer=None,
            start_frame_index = None,
    ):
        """
        A gym-like interface for simulating traffic behaviors (both ego and other agents) with L5Kit's SimulationDataset

        Args:
            env_config (L5KitEnvConfig): a Config object specifying the behavior of the L5Kit CloseLoopSimulator
            num_scenes (int): number of scenes to run in parallel
            dataset (EgoDataset): an EgoDataset instance that contains scene data for simulation
            prediction_only (bool): if set to True, ignore the input action command and only record the predictions
        """
        if start_frame_index is None:
            start_frame_index = env_config.simulation.start_frame_index
        self._sim_cfg = SimulationConfig(
            disable_new_agents=True,
            distance_th_far=env_config.simulation.distance_th_far,
            distance_th_close=env_config.simulation.distance_th_close,
            num_simulation_steps=env_config.simulation.num_simulation_steps,
            start_frame_index=start_frame_index,
            show_info=True,
        )

        self.generate_agent_obs = env_config.get("generate_agent_obs", True)
        self._npr = np.random.RandomState(seed=seed)
        self.dataset = dataset

        if renderer is not None:
            self.render_rasterizer = renderer
        else:
            self.render_rasterizer = dataset.rasterizer

        self._num_total_scenes = len(dataset.dataset.scenes)
        self._num_scenes = num_scenes

        # indices of the scenes (in dataset) that are being used for simulation
        self._current_scene_indices = None
        self._current_scene_dataset = None  # corresponding dataset of the scenes
        # agent IDs of the current episode for each scene
        self._current_agent_track_ids = None

        self._frame_index = 0
        self._cached_observation = None
        self._done = False
        self._prediction_only = prediction_only

        self.logger = None
        self.gt_logger = None

        self.timers = Timers()

        self._metrics = dict() if metrics is None else metrics
        self._skimp = skimp_rollout

    def update_random_seed(self,seed):
        self._npr = np.random.RandomState(seed=seed)
    
    def reset_multi_episodes_metrics(self):
        for v in self._metrics.values():
            v.multi_episode_reset()

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
        else:
            scene_indices = np.array(scene_indices)
            self._num_scenes = len(scene_indices)

        assert (
            np.max(scene_indices) < self._num_total_scenes
            and np.min(scene_indices) >= 0
        )

        if start_frame_index is not None:
            self._sim_cfg.start_frame_index = start_frame_index

        self._current_scene_indices = scene_indices
        self._current_scene_dataset = SimulationDataset.from_dataset_indices(
            self.dataset, scene_indices, self._sim_cfg
        )
        self._current_agent_track_ids = None

        for scene_ds in self._current_scene_dataset.recorded_scene_dataset_batch.values():
            scene_ds.set_skimp(True)

        self._frame_index = 0
        self._cached_observation = None
        self._done = False

        for v in self._metrics.values():
            v.reset()

        obs_keys_to_log = [
            "centroid",
            "yaw",
            "curr_speed",
            "extent",
            "raster_from_agent",
            "world_from_agent",
            "raster_from_world",
            "scene_index",
            "track_id",
        ]
        self.logger = RolloutLogger(obs_keys=obs_keys_to_log)
        self.gt_logger = RolloutLogger(obs_keys=obs_keys_to_log)

    def get_random_action(self):
        ac = self._npr.randn(self._num_scenes, 1, 3)
        ego = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )
        ac = self._npr.randn(len(self._current_agent_track_ids), 1, 3)
        agents = Action(
            positions=ac[:, :, :2],
            yaws=ac[:, :, 2:3]
        )

        return RolloutAction(ego=ego, agents=agents)

    def get_gt_action(self, obs):
        ego = Action(
            positions=obs["ego"]["target_positions"],
            yaws=obs["ego"]["target_yaws"]
        )
        agents = Action(
            positions=obs["agents"]["target_positions"],
            yaws=obs["agents"]["target_yaws"]
        )
        return RolloutAction(ego=ego, agents=agents)

    @property
    def current_scene_indices(self):
        return self._current_scene_indices.copy()

    @property
    def num_instances(self):
        return self._num_scenes

    @property
    def total_num_scenes(self):
        return self._num_total_scenes

    def get_info(self):
        sim_buffer = self.logger.get_serialized_scene_buffer()
        gt_buffer = self.gt_logger.get_serialized_scene_buffer()
        joint_buffer = sim_buffer
        for si in gt_buffer:
            for k in gt_buffer[si]:
                joint_buffer[si]["gt_{}".format(k)] = gt_buffer[si][k]

        joint_buffer = [joint_buffer[k] for k in self.current_scene_indices]

        return {
            "buffer": joint_buffer,
            "scene_index": self.current_scene_indices,
            "experience": self.get_episode_experience()
        }

    @property
    def agents_to_scene_index(self):
        """Get the corresponding scene index (start from 0) for each agent in the obs"""
        obs = self.get_observation()
        scene_index = obs["agents"]["scene_index"].copy()
        for i, s in enumerate(scene_index):
            scene_index[i] = np.where(self.current_scene_indices == s)[0]
        return scene_index

    @property
    def scene_to_agents_index(self):
        """Return a list of agent indices for each scene"""
        obs = self.get_observation()
        scene_index = obs["agents"]["scene_index"]
        agent_indices = []
        for si in self.current_scene_indices:
            agent_indices.append(np.where(scene_index == si)[0])
        return agent_indices

    @property
    def scene_to_ego_index(self):
        return np.split(np.arange(self.num_instances), self.num_instances)

    def get_multi_episode_metrics(self):
        metrics = dict()
        for met_name, met in self._metrics.items():
            met_vals = met.get_multi_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            elif met_vals is not None:
                metrics[met_name] = met_vals
        return metrics

    def get_metrics(self):
        """
        Get metrics of the current episode (may compute before is_done==True)

        Returns: a dictionary of metrics, each containing an array of measurement same length as the number of scenes
        """

        # make sure the agent ordering is consistent
        for si in self.current_scene_indices:
            assert np.all(self.logger.get_track_id()[si] == self.gt_logger.get_track_id()[si])

        sim_traj = self.logger.get_trajectory()
        gt_traj = self.gt_logger.get_trajectory()
        ade = np.zeros(self.num_instances)
        fde = np.zeros(self.num_instances)
        for i, si in enumerate(self.current_scene_indices):
            traj_dist = np.linalg.norm(sim_traj[si]["positions"] - gt_traj[si]["positions"], axis=-1)
            ade[i] = traj_dist.mean()
            fde[i] = traj_dist[:, -1].mean()

        metrics = {
            "ADE": ade,
            "FDE": fde,
        }
        # aggregate per-step metrics
        # self._add_per_step_metrics(self.get_observation())
        for met_name, met in self._metrics.items():

            met_vals = met.get_episode_metrics()
            if isinstance(met_vals, dict):
                for k, v in met_vals.items():
                    metrics[met_name + "_" + k] = v
            elif met_vals is not None:
                metrics[met_name] = met_vals

        return metrics

    def get_observation(self):
        if self._done:
            return None

        if self._cached_observation is not None:
            return self._cached_observation

        self.timers.tic("get_obs")
        # Get agent observations
        if self.generate_agent_obs:
            agent_obs = self._current_scene_dataset.rasterise_agents_frame_batch(
                self._frame_index)

            if len(agent_obs) > 0:
                agent_obs = default_collate(list(agent_obs.values()))
        else:
            agent_obs = None
        # Get ego observation
        ego_obs = self._current_scene_dataset.rasterise_frame_batch(
            self._frame_index)
        ego_obs = default_collate(ego_obs)

        # cache observations
        self._cached_observation = TensorUtils.to_numpy(
            {"agents": agent_obs, "ego": ego_obs}
        )
        self.timers.toc("get_obs")

        return self._cached_observation

    def get_observation_gt(self):
        if self.generate_agent_obs:
            agent_obs = self._current_scene_dataset.rasterise_agents_frame_batch_gt(
                self._frame_index)

            if len(agent_obs) > 0:
                agent_obs = default_collate(list(agent_obs.values()))
        else:
            agent_obs = None
        # Get ego observation
        ego_obs = self._current_scene_dataset.rasterise_frame_batch_gt(
            self._frame_index)
        ego_obs = default_collate(ego_obs)

        # cache observations
        gt_obs = TensorUtils.to_numpy(
            {"agents": agent_obs, "ego": ego_obs}
        )
        return gt_obs

    def _get_observation_by_index(self, scene_index, frame_index, agent_track_ids, collate=True):
        agents_dict = OrderedDict()
        dataset = self._current_scene_dataset.scene_dataset_batch[scene_index]
        frame = dataset.dataset.frames[frame_index]
        frame_agents = filter_agents_by_frames(
            frame, dataset.dataset.agents)[0]
        for agent in frame_agents:
            track_id = int(agent["track_id"])
            if track_id in agent_track_ids:
                el = dataset.get_frame(
                    scene_index=0, state_index=frame_index, track_id=track_id
                )
                # we replace the scene_index here to match the real one (otherwise is 0)
                el["scene_index"] = scene_index
                agents_dict[scene_index, track_id] = el
        if collate:
            agents_dict = default_collate(list(agents_dict.values()))
            return TensorUtils.to_numpy(agents_dict)
        else:
            return agents_dict

    def combine_observations(self, obs):
        """Get a combined observation by concatenating ego and agents"""
        if obs is None:
            return None

        combined = dict(obs["ego"])
        if obs["agents"] is not None:
            for k in obs["ego"].keys():
                combined[k] = np.concatenate(
                    (obs["ego"][k], obs["agents"][k]), axis=0)
        return combined

    def is_done(self):
        return self._done

    def get_reward(self):
        # TODO
        return np.zeros(self._num_scenes)

    def render(self, actions_to_take: RolloutAction, **kwargs):
        ims = []
        metrics_to_vis = dict()
        for i, si in enumerate(self.current_scene_indices):
            im = render_state_l5kit_ego_view(
                rasterizer=self.render_rasterizer,
                state_obs=self.get_observation(),
                action=actions_to_take,
                step_index=self._frame_index,
                dataset_scene_index=si,
                step_metrics=TensorUtils.map_ndarray(
                    metrics_to_vis, lambda x: x[i]),
                **kwargs
            )
            ims.append(im)
        ims = np.stack(ims)
        return ims

    @property
    def horizon(self):
        return len(self._current_scene_dataset)

    def _add_per_step_metrics(self, obs):
        for k, v in self._metrics.items():
            if k.startswith("ego"):
                v.add_step(obs["ego"], self.current_scene_indices)
            elif k.startswith("agents"):
                v.add_step(obs["agents"], self.current_scene_indices)
            elif k.startswith("all"):
                v.add_step(self.combine_observations(obs), self.current_scene_indices)
            else:
                raise KeyError("Invalid metrics name {}".format(k))

    def _step(self, step_actions: RolloutAction):
        obs = self.get_observation()
        # record metrics
        # self._add_per_step_metrics(obs)

        # record observations and actions
        self.logger.log_step(obs, step_actions)
        gt_obs = self.get_observation_gt()
        self.gt_logger.log_step(gt_obs, self.get_gt_action(gt_obs))

        should_update = self._frame_index + 1 < self.horizon and not self._prediction_only
        self.timers.tic("update")
        if step_actions.has_ego:
            if should_update:
                # update the next frame's ego position and orientation using control input
                ClosedLoopSimulator.update_ego(
                    dataset=self._current_scene_dataset,
                    frame_idx=self._frame_index + 1,
                    input_dict=obs["ego"],
                    output_dict=step_actions.ego.to_dict(),
                )

        if step_actions.has_agents:
            if should_update:
                # update the next frame's agent positions and orientations using control input
                ClosedLoopSimulator.update_agents(
                    dataset=self._current_scene_dataset,
                    frame_idx=self._frame_index + 1,
                    input_dict=obs["agents"],
                    output_dict=step_actions.agents.to_dict(),
                )

        self.timers.toc("update")

        # TODO: accumulate sim trajectories
        self._cached_observation = None
        if self._frame_index + 1 == self.horizon:
            self._done = True
        else:
            self._frame_index += 1

    def set_dataset_skimp_mode(self, skimp):
        for scene_ds in self._current_scene_dataset.scene_dataset_batch.values():
            scene_ds.set_skimp(skimp)

    def step(self, actions: RolloutAction, num_steps_to_take: int = 1, render=False):
        """
        Step the simulation with control inputs

        Args:
            actions (RolloutAction): action for controlling ego and/or agents
            num_steps_to_take (int): how many env steps to take. Must be less or equal to length of the input actions
            render (bool): whether to render state and actions and return renderings
        """

        self.set_dataset_skimp_mode(self._skimp)
        if self.is_done():
            raise SimulationException("Simulation episode has ended")

        # otherwise, use @actions to update simulation
        actions = actions.to_numpy()
        # Convert ego actions to world frame
        obs = self.get_observation()
        self._add_per_step_metrics(obs)
        
        actions_world = actions.transform(
            ego_trans_mats=obs["ego"]["world_from_agent"],
            ego_rot_rads=obs["ego"]["yaw"][..., None, None],
            agents_trans_mats=obs["agents"]["world_from_agent"] if self.generate_agent_obs else None,
            agents_rot_rads=obs["agents"]["yaw"][..., None,
                                                 None] if self.generate_agent_obs else None
        )
        if actions.has_agents:
            agent_track_ids = obs["agents"]["track_id"]
            agent_scene_indices = obs["agents"]["scene_index"]

        renderings = []
        for step_i in range(num_steps_to_take):
            if self.is_done():
                break

            obs = self.get_observation()
            
            actions_world_d = actions_world.to_dict()
            if actions.has_agents:
                # some agents might get dropped in the middle,
                # index actions by the current agent track ids and scene index
                active_agent_index = np.zeros(obs["agents"]["scene_index"].shape[0], dtype=np.int64)
                newly_added = list()
                for i, (tid, sid) in enumerate(zip(obs["agents"]["track_id"], obs["agents"]["scene_index"])):
                    action_index = np.bitwise_and(tid == agent_track_ids, sid == agent_scene_indices)
                    if action_index.any():
                        active_agent_index[i] = np.where(action_index)[0][0]
                    else:
                        # accounting for newly added agents that do not have a rollout action
                        active_agent_index[i] = i
                        newly_added.append(i)

                actions_world_d["agents"] = TensorUtils.map_ndarray(
                    actions_world_d["agents"], lambda x: x[active_agent_index]
                )
                actions_world_d["agents_info"] = TensorUtils.map_ndarray(
                    actions_world_d["agents_info"], lambda x: x[active_agent_index]
                )

            actions_world_d["agents"] = TensorUtils.map_ndarray(actions_world_d["agents"], lambda x: x[:, step_i:])
            actions_world_d["ego"] = TensorUtils.map_ndarray(actions_world_d["ego"], lambda x: x[:, step_i:])

            if actions.has_agents:
                if len(newly_added)>0:
                    T = actions_world_d["agents"]["positions"].shape[1]
                    actions_world_d["agents"]["positions"][newly_added,:T] = obs["agents"]["target_positions"][newly_added,:T]
                    actions_world_d["agents"]["yaws"][newly_added,:T] = obs["agents"]["target_yaws"][newly_added,:T]

            step_actions_world = RolloutAction.from_dict(actions_world_d)

            step_actions = step_actions_world.transform(
                ego_trans_mats=obs["ego"]["agent_from_world"],
                ego_rot_rads=- obs["ego"]["yaw"][..., None, None],
                agents_trans_mats=obs["agents"]["agent_from_world"] if self.generate_agent_obs else None,
                agents_rot_rads=- obs["agents"]["yaw"][..., None,
                                                       None] if self.generate_agent_obs else None
            )

            if render:
                if step_actions.ego_info is not None and "plan_info" in step_actions.ego_info:
                    step_actions.ego_info["plan_info"] = actions.ego_info["plan_info"]
                renderings.append(self.render(actions_to_take=step_actions))

            self._step(step_actions=step_actions)

        self.set_dataset_skimp_mode(False)
        return renderings

    def get_episode_experience(self):
        """Get episodic experience in the form of datasets"""
        ds = []
        for si in self.current_scene_indices:
            ds.append(
                (si, self._current_scene_dataset.scene_dataset_batch[si].dataset))
        return ds

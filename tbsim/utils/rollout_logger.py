from collections import defaultdict
import numpy as np
from copy import deepcopy

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.common import RolloutAction

from torch.nn.utils.rnn import pad_sequence
class RolloutLogger(object):
    """Log trajectories and other essential info during rollout for visualization and evaluation"""
    def __init__(self, obs_keys=None):
        if obs_keys is None:
            obs_keys = dict()
        self._obs_keys = obs_keys
        self._scene_indices = None
        self._agent_id_per_scene = dict()
        self._agent_data_by_scene = dict()
        self._scene_ts = defaultdict(lambda:0)

    def _combine_obs(self, obs):
        combined = dict()
        excluded_keys = ["extras"]
        if "ego" in obs and obs["ego"] is not None:
            combined.update(obs["ego"])
        if "agents" in obs and obs["agents"] is not None:
            for k in obs["agents"].keys():
                if k in combined and k not in excluded_keys:
                    if obs["agents"][k] is not None:
                        if combined[k] is not None:
                            combined[k] = np.concatenate((combined[k], obs["agents"][k]), axis=0)
                        else:
                            combined[k] = obs["agents"][k]
                else:
                    combined[k] = obs["agents"][k]
        return combined

    def _combine_action(self, action: RolloutAction):
        combined = dict(action=dict())
        if action.has_ego and not action.has_agents:
            combined["action"] = action.ego.to_dict()
            if action.ego_info is not None and "action_samples" in action.ego_info:
                combined["action_samples"] = action.ego_info["action_samples"]
            return combined
                
        elif action.has_agents and not action.has_ego:
            combined["action"] = action.agents.to_dict()
            if action.agents_info is not None and "action_samples" in action.agents_info:
                combined["action_samples"] = action.agents_info["action_samples"]
            return combined
        elif action.has_agents and action.has_ego:
            Nego = action.ego.positions.shape[0]
            Nagents = action.agents.positions.shape[0]
            combined["action"] = dict()
            agents_action = action.agents.to_dict()
            ego_action = action.ego.to_dict()
            for k in agents_action:
                if k in ego_action:
                    combined["action"][k] = np.concatenate((ego_action[k], agents_action[k]), axis=0)
            if action.agents_info is not None and action.ego_info is not None:
                if "action_samples" in action.ego_info:
                    ego_samples = action.ego_info["action_samples"]
                else:
                    ego_samples = None
                if "action_samples" in action.agents_info:
                    agents_samples = action.agents_info["action_samples"]
                else:
                    agents_samples = None
                if ego_samples is not None and agents_samples is None:
                    combined["action_samples"] = dict()
                    for k in ego_samples:
                        pad_k = np.zeros([Nagents,*ego_samples[k].shape[1:]])
                        combined["action_samples"][k]=np.concatenate((ego_samples[k],pad_k),0)
                elif ego_samples is None and agents_samples is not None:
                    combined["action_samples"] = dict()
                    for k in agents_samples:
                        pad_k = np.zeros([Nego,*agents_samples[k].shape[1:]])
                        combined["action_samples"][k]=np.concatenate((pad_k,agents_samples[k]),0)
                elif ego_samples is not None and agents_samples is not None:
                    combined["action_samples"] = dict()
                    for k in ego_samples:
                        if k in agents_samples:
                            if ego_samples[k].shape[1]>agents_samples[k].shape[1]:
                                pad_k = np.zeros([Nagents,ego_samples[k].shape[1]-agents_samples[k].shape[1],*agents_samples[k].shape[2:]])
                                agents_samples[k]=np.concatenate((agents_samples[k],pad_k),1)
                            elif ego_samples[k].shape[1]<agents_samples[k].shape[1]:
                                pad_k = np.zeros([Nego,agents_samples[k].shape[1]-ego_samples[k].shape[1],*ego_samples[k].shape[2:]])
                                ego_samples[k]=np.concatenate((ego_samples[k],pad_k),1)
                            combined["action_samples"][k] = np.concatenate((ego_samples[k],agents_samples[k]),0)
        return combined

    def _maybe_initialize(self, obs):
        if self._scene_indices is None:
            self._scene_indices = np.unique(obs["scene_index"])
            self._scene_ts = defaultdict(lambda:0)
            for si in self._scene_indices:
                self._agent_id_per_scene[si] = obs["track_id"][obs["scene_index"] == si]
            for si in self._scene_indices:
                self._agent_data_by_scene[si] = defaultdict(lambda:defaultdict(lambda:dict()))

    def _append_buffer(self, obs, action):
        """
        scene_index:
            dict(
                action_positions=[[num_agent, ...], [num_agent, ...], ],
                action_yaws=[[num_agent, ...], [num_agent, ...], ],
                centroid=[[num_agent, ...], [num_agent, ...], ],
                ...
            )
        """
        self._serialized_scene_buffer = None  # need to re-serialize

        # TODO: move this to __init__ as arg
        state = {k: obs[k] for k in self._obs_keys}
        state["action_positions"] = action["action"]["positions"]
        state["action_yaws"] = action["action"]["yaws"]
        if "action_samples" in action:
            # only collect up to 20 samples to save space
            # state["action_sample_positions"] = action["action_samples"]["positions"][:, :20]
            # state["action_sample_yaws"] = action["action_samples"]["yaws"][:, :20]

            state["action_sample_positions"] = action["action_samples"]["positions"]
            state["action_sample_yaws"] = action["action_samples"]["yaws"]
        
        for si in self._scene_indices:
            self._agent_id_per_scene[si] = obs["track_id"][obs["scene_index"] == si]
            scene_mask = np.where(si == obs["scene_index"])[0]
            scene_state = TensorUtils.map_ndarray(state, lambda x: x[scene_mask])
            ts = self._scene_ts[si]
            scene_track_id = scene_state["track_id"]
            for i, ti in enumerate(scene_track_id):
                for k in scene_state:
                    self._agent_data_by_scene[si][k][ti][ts] = scene_state[k][i:i+1]
            

    def get_serialized_scene_buffer(self):
        """
        scene_index:
            dict(
                action_positions=[num_agent, T, ...],
                action_yaws=[num_agent, T, ...],
                centroid=[num_agent, T, ...],
                ...
            )
        """

        if self._serialized_scene_buffer is not None:
            return self._serialized_scene_buffer

        serialized = dict()
        for si in self._agent_data_by_scene:
            serialized[si] = dict()
            for k in self._agent_data_by_scene[si]:
                serialized[si][k]=list()
                for ti in self._agent_data_by_scene[si][k]:
                    if len(self._agent_data_by_scene[si][k][ti])>0:
                        default_val = list(self._agent_data_by_scene[si][k][ti].values())[0]
                        ti_k = list()
                        for ts in range(self._scene_ts[si]):
                            ti_k.append(self._agent_data_by_scene[si][k][ti][ts] if ts in self._agent_data_by_scene[si][k][ti] else np.ones_like(default_val)*np.nan)
                            default_val = ti_k[-1]
                        if not all(elem.shape==ti_k[0].shape for elem in ti_k):
                            # requires padding
                            if np.issubdtype(ti_k[0].dtype,float):
                                padding_value = np.nan
                            else:
                                padding_value = 0
                            ti_k = [x[0] for x in ti_k]
                            ti_k_torch = TensorUtils.to_tensor(ti_k,ignore_if_unspecified=True)

                            ti_k_padded = pad_sequence(ti_k_torch,padding_value=padding_value,batch_first=True)
                            serialized[si][k].append(TensorUtils.to_numpy(ti_k_padded)[np.newaxis,:])
                        else:
                            if ti_k[0].ndim==0:
                                serialized[si][k].append(np.array(ti_k)[np.newaxis,:])
                            else:
                                serialized[si][k].append(np.concatenate(ti_k,axis=0)[np.newaxis,:])
                    else:
                        serialized[si][k].append(np.zeros_like(serialized[si][k][-1]))
                if not all(elem.shape==serialized[si][k][0].shape for elem in serialized[si][k]):
                    # requires padding
                    if np.issubdtype(serialized[si][k][0][0].dtype,float):
                        padding_value = np.nan
                    else:
                        padding_value = 0
                    axes=[1,0]+np.arange(2,serialized[si][k][0].ndim-1).tolist()
                    mk_transpose = [np.transpose(x[0],axes) for x in serialized[si][k]]
                    mk_torch = TensorUtils.to_tensor(mk_transpose,ignore_if_unspecified=True)
                    mk_padded = pad_sequence(mk_torch,padding_value=padding_value)
                    mk = TensorUtils.to_numpy(mk_padded)
                    axes=[1,2,0]+np.arange(3,mk.ndim).tolist()
                    serialized[si][k]=np.transpose(mk,axes)
                else:
                    serialized[si][k] = np.concatenate(serialized[si][k],axis=0)

                

        self._serialized_scene_buffer = serialized
        return deepcopy(self._serialized_scene_buffer)

    def get_trajectory(self):
        """Get per-scene rollout trajectory in the world coordinate system"""
        buffer = self.get_serialized_scene_buffer()
        traj = dict()
        for si in buffer:
            traj[si] = dict(
                positions=buffer[si]["centroid"],
                yaws=buffer[si]["yaw"]
            )
        return traj

    def get_track_id(self):
        return deepcopy(self._agent_id_per_scene)

    def get_stats(self):
        # TODO
        raise NotImplementedError()

    def log_step(self, obs, action: RolloutAction):
        combined_obs = self._combine_obs(obs)
        combined_action = self._combine_action(action)
        assert combined_obs["scene_index"].shape[0] == combined_action["action"]["positions"].shape[0]
        self._maybe_initialize(combined_obs)
        self._append_buffer(combined_obs, combined_action)
        for si in np.unique(combined_obs["scene_index"]):
            self._scene_ts[si]+=1
        del combined_obs

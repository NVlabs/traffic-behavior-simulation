import numpy as np
import torch
from copy import deepcopy

import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.geometry_utils import transform_points_tensor
from l5kit.geometry import transform_points


class Trajectory(object):
    """Container for sequences of 2D positions and yaws"""
    def __init__(self, positions, yaws):
        assert positions.shape[:-1] == yaws.shape[:-1]
        assert positions.shape[-1] == 2
        assert yaws.shape[-1] == 1
        self._positions = positions
        self._yaws = yaws

    @property
    def trajectories(self):
        if isinstance(self.positions, np.ndarray):
            return np.concatenate([self._positions, self._yaws], axis=-1)
        else:
            return torch.cat([self._positions, self._yaws], dim=-1)

    @property
    def positions(self):
        return TensorUtils.clone(self._positions)

    @positions.setter
    def positions(self, x):
        self._positions = TensorUtils.clone(x)

    @property
    def yaws(self):
        return TensorUtils.clone(self._yaws)

    @yaws.setter
    def yaws(self, x):
        self._yaws = TensorUtils.clone(x)

    def to_dict(self):
        return dict(
            positions=self.positions,
            yaws=self.yaws
        )

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def transform(self, trans_mats, rot_rads):
        if isinstance(self.positions, np.ndarray):
            pos = transform_points(self.positions, trans_mats)
        else:
            pos = transform_points_tensor(self.positions, trans_mats)

        yaw = self.yaws + rot_rads
        return self.__class__(pos, yaw)

    def to_numpy(self):
        return self.__class__(**TensorUtils.to_numpy(self.to_dict()))


class Action(Trajectory):
    pass


class Plan(Trajectory):
    """Container for sequences of 2D positions, yaws, controls, availabilities."""
    def __init__(self, positions, yaws, availabilities, controls=None):
        assert positions.shape[:-1] == yaws.shape[:-1]
        assert positions.shape[-1] == 2
        assert yaws.shape[-1] == 1
        assert availabilities.shape == positions.shape[:-1]
        self._positions = positions
        self._yaws = yaws
        self._availabilities = availabilities
        self._controls = controls

    @property
    def availabilities(self):
        return TensorUtils.clone(self._availabilities)

    @property
    def controls(self):
        return TensorUtils.clone(self._controls)

    def to_dict(self):
        p = dict(
            positions=self.positions,
            yaws=self.yaws,
            availabilities=self.availabilities,
        )
        if self._controls is not None:
            p["controls"] = self.controls
        return p

    def transform(self, trans_mats, rot_rads):
        if isinstance(self.positions, np.ndarray):
            pos = transform_points(self.positions, trans_mats)
        else:
            pos = transform_points_tensor(self.positions, trans_mats)

        yaw = self.yaws + rot_rads
        return self.__class__(pos, yaw, self.availabilities, controls=self.controls)


class RolloutAction(object):
    """Actions used to control agent rollouts"""
    def __init__(self, ego=None, ego_info=None, agents=None, agents_info=None):
        assert ego is None or isinstance(ego, Action)
        assert agents is None or isinstance(agents, Action)
        assert ego_info is None or isinstance(ego_info, dict)
        assert agents_info is None or isinstance(agents_info, dict)

        self.ego = ego
        self.ego_info = ego_info
        self.agents = agents
        self.agents_info = agents_info

    @property
    def has_ego(self):
        return self.ego is not None

    @property
    def has_agents(self):
        return self.agents is not None

    def transform(self, ego_trans_mats, ego_rot_rads, agents_trans_mats=None, agents_rot_rads=None):
        trans_action = RolloutAction()
        if self.has_ego:
            trans_action.ego = self.ego.transform(
                trans_mats=ego_trans_mats, rot_rads=ego_rot_rads)
            if self.ego_info is not None:
                trans_action.ego_info = deepcopy(self.ego_info)
                if "plan" in trans_action.ego_info:
                    plan = Plan.from_dict(trans_action.ego_info["plan"])
                    trans_action.ego_info["plan"] = plan.transform(
                        trans_mats=ego_trans_mats, rot_rads=ego_rot_rads
                    ).to_dict()
        if self.has_agents:
            assert agents_trans_mats is not None and agents_rot_rads is not None
            trans_action.agents = self.agents.transform(
                trans_mats=agents_trans_mats, rot_rads=agents_rot_rads)
            if self.agents_info is not None:
                trans_action.agents_info = deepcopy(self.agents_info)
                if "plan" in trans_action.agents_info:
                    plan = Plan.from_dict(trans_action.agents_info["plan"])
                    trans_action.agents_info["plan"] = plan.transform(
                        trans_mats=agents_trans_mats, rot_rads=agents_rot_rads
                    ).to_dict()
        return trans_action

    def to_dict(self):
        d = dict()
        if self.has_ego:
            d["ego"] = self.ego.to_dict()
            d["ego_info"] = deepcopy(self.ego_info)
        if self.has_agents:
            d["agents"] = self.agents.to_dict()
            d["agents_info"] = deepcopy(self.agents_info)
        return d

    def to_numpy(self):
        return self.__class__(
            ego=self.ego.to_numpy() if self.has_ego else None,
            ego_info=TensorUtils.to_numpy(
                self.ego_info) if self.has_ego else None,
            agents=self.agents.to_numpy() if self.has_agents else None,
            agents_info=TensorUtils.to_numpy(
                self.agents_info) if self.has_agents else None,
        )

    @classmethod
    def from_dict(cls, d):
        d = deepcopy(d)
        if "ego" in d:
            d["ego"] = Action.from_dict(d["ego"])
        if "agents" in d:
            d["agents"] = Action.from_dict(d["agents"])
        return cls(**d)


from tbsim.dynamics.base import DynType, Dynamics
from tbsim.utils.math_utils import soft_sat
import torch
import numpy as np
from copy import deepcopy



class DoubleIntegrator(Dynamics):
    def __init__(self, name, abound, vbound=None):
        self._name = name
        self._type = DynType.DI
        self.xdim = 2
        self.udim = 2
        self.cyclic_state = list()
        self.vbound = vbound
        self.abound = abound

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            return np.hstack((x[..., 2:], u))
        elif isinstance(x, torch.Tensor):
            return torch.cat((x[..., 2:], u), dim=-1)
        else:
            raise NotImplementedError

    def step(self, x, u, dt, bound=True):

        if isinstance(x, np.ndarray):
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)
            xn = np.hstack(
                ((x[..., 2:4] + 0.5 * u * dt) * dt + x[..., 0:2], x[..., 2:4] + u * dt)
            )
        elif isinstance(x, torch.Tensor):
            if bound:
                lb, ub = self.ubound(x)
                u = torch.clip(u, min=lb, max=ub)
            xn = torch.clone(x)
            xn[..., 0:2] += (x[..., 2:4] + 0.5 * u * dt) * dt
            xn[..., 2:4] += u * dt
        else:
            raise NotImplementedError
        return xn

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if self.vbound is None:
            if isinstance(x, np.ndarray):
                lb = np.ones_like(x[..., 2:]) * self.abound[0]
                ub = np.ones_like(x[..., 2:]) * self.abound[1]

            elif isinstance(x, torch.Tensor):
                lb = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 0]
                ).to(x.device)
                ub = torch.ones_like(x[..., 2:]) * torch.from_numpy(
                    self.abound[:, 1]
                ).to(x.device)

            else:
                raise NotImplementedError
        else:
            if isinstance(x, np.ndarray):
                lb = (x[..., 2:] > self.vbound[0]) * self.abound[0]
                ub = (x[..., 2:] < self.vbound[1]) * self.abound[1]

            elif isinstance(x, torch.Tensor):
                lb = (
                    x[..., 2:] > torch.from_numpy(self.vbound[0]).to(x.device)
                ) * torch.from_numpy(self.abound[0]).to(x.device)
                ub = (
                    x[..., 2:] < torch.from_numpy(self.vbound[1]).to(x.device)
                ) * torch.from_numpy(self.abound[1]).to(x.device)
            else:
                raise NotImplementedError
        return lb, ub

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        # return torch.atan2(x[..., 3:], x[..., 2:3])
        return torch.zeros_like(x[..., 0:1])
    @staticmethod
    def inverse_dyn(x,xp,dt):
        return (xp[...,2:]-x[...,2:])/dt
    @staticmethod
    def calculate_vel(pos, yaw, dt, mask):
        vel = (pos[...,1:,:]-pos[...,:-1,:])/dt
        if isinstance(pos, torch.Tensor):
            # right finite difference velocity
            vel_r = torch.cat((vel[..., 0:1, :], vel), dim=-2)
            # left finite difference velocity
            vel_l = torch.cat((vel, vel[..., -1:, :]), dim=-2)
            mask_r = torch.roll(mask, 1, dims=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask

            mask_l = torch.roll(mask, -1, dims=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                (mask_l & mask_r).unsqueeze(-1) * (vel_r + vel_l) / 2
                + (mask_l & (~mask_r)).unsqueeze(-1) * vel_l
                + (mask_r & (~mask_l)).unsqueeze(-1) * vel_r
            )
        elif isinstance(pos, np.ndarray):
            # right finite difference velocity
            vel_r = np.concatenate((vel[..., 0:1, :], vel), axis=-2)
            # left finite difference velocity
            vel_l = np.concatenate((vel, vel[..., -1:, :]), axis=-2)
            mask_r = np.roll(mask, 1, axis=-1)
            mask_r[..., 0] = False
            mask_r = mask_r & mask
            mask_l = np.roll(mask, -1, axis=-1)
            mask_l[..., -1] = False
            mask_l = mask_l & mask
            vel = (
                np.expand_dims(mask_l & mask_r,-1) * (vel_r + vel_l) / 2
                + np.expand_dims(mask_l & (~mask_r),-1) * vel_l
                + np.expand_dims(mask_r & (~mask_l),-1) * vel_r
            )
        else:
            raise NotImplementedError
        return vel
    @staticmethod
    def get_state(pos,yaw,dt,mask):
        vel = DoubleIntegrator.calculate_vel(pos, yaw, dt, mask)
        if isinstance(vel,np.ndarray):
            return np.concatenate((pos,vel),-1)
        elif isinstance(vel,torch.Tensor):
            return torch.cat((pos,vel),-1)

    def forward_dynamics(self,
                         initial_states: torch.Tensor,
                         actions: torch.Tensor,
                         step_time: float,
                        ):
        if isinstance(actions, np.ndarray):
            actions = np.clip(actions,self.abound[0],self.abound[1])
            delta_v = np.cumsum(actions*step_time,-2)
            vel = initial_states[...,np.newaxis,2:]+delta_v
            vel = np.clip(vel,self.vbound[0],self.vbound[1])
            delta_xy = np.cumsum(vel*step_time,-2)
            xy = initial_states[...,np.newaxis,:2]+delta_xy

            traj = np.concatenate((xy,vel),-1)
        elif isinstance(actions,torch.Tensor):
            actions = soft_sat(actions,self.abound[0],self.abound[1])
            delta_v = torch.cumsum(actions*step_time,-2)
            vel = initial_states[...,2:].unsqueeze(-2)+delta_v
            vel = soft_sat(vel,self.vbound[0],self.vbound[1])
            delta_xy = torch.cumsum(vel*step_time,-2)
            xy = initial_states[...,:2].unsqueeze(-2)+delta_xy

            traj = torch.cat((xy,vel),-1)
        xy = self.state2pos(traj)
        yaw = self.state2yaw(traj)
        return traj, xy, yaw
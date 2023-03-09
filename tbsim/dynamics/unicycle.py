from tbsim.dynamics.base import DynType, Dynamics
from tbsim.utils.math_utils import soft_sat
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.autograd.functional import jacobian



class Unicycle(Dynamics):
    def __init__(
        self, name = None, max_steer=0.5, max_yawvel=8, acce_bound=[-6, 4], vbound=[-10, 30]
    ):
        self._name = name
        self._type = DynType.UNICYCLE
        self.xdim = 4
        self.udim = 2
        self.cyclic_state = [3]
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.max_steer = max_steer
        self.max_yawvel = max_yawvel

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            theta = x[..., 3:4]
            dxdt = np.hstack(
                (np.cos(theta) * x[..., 2:3], np.sin(theta) * x[..., 2:3], u)
            )
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            theta = x[..., 3:4]
            dxdt = torch.cat(
                (torch.cos(theta) * x[..., 2:3],
                 torch.sin(theta) * x[..., 2:3], u),
                dim=-1,
            )
        else:
            raise NotImplementedError
        return dxdt

    def step(self, x, u, dt, bound=True):
        assert x.shape[:-1] == u.shape[:-1]
        if isinstance(x, np.ndarray):
            assert isinstance(u, np.ndarray)
            if bound:
                lb, ub = self.ubound(x)
                u = np.clip(u, lb, ub)

            theta = x[..., 3:4]
            dxdt = np.hstack(
                (
                    np.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    np.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                )
            )
        elif isinstance(x, torch.Tensor):
            assert isinstance(u, torch.Tensor)
            if bound:
                lb, ub = self.ubound(x)
                # s = (u - lb) / torch.clip(ub - lb, min=1e-3)
                # u = lb + (ub - lb) * torch.sigmoid(s)
                u = torch.clip(u, lb, ub)

            theta = x[..., 3:4]
            dxdt = torch.cat(
                (
                    torch.cos(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    torch.sin(theta) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                    u,
                ),
                dim=-1,
            )
        else:
            raise NotImplementedError
        return x + dxdt * dt

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if isinstance(x, np.ndarray):
            v = x[..., 2:3]
            vclip = np.clip(np.abs(v), a_min=0.1, a_max=None)
            yawbound = np.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            acce_lb = np.clip(
                np.clip(self.vbound[0] - v, None, self.acce_bound[1]),
                self.acce_bound[0],
                None,
            )
            acce_ub = np.clip(
                np.clip(self.vbound[1] - v, self.acce_bound[0], None),
                None,
                self.acce_bound[1],
            )
            lb = np.concatenate((acce_lb, -yawbound),-1)
            ub = np.concatenate((acce_ub, yawbound),-1)
            return lb, ub
        elif isinstance(x, torch.Tensor):
            v = x[..., 2:3]
            vclip = torch.clip(torch.abs(v),min=0.1)
            yawbound = torch.minimum(
                self.max_steer * vclip,
                self.max_yawvel / vclip,
            )
            yawbound = torch.clip(yawbound, min=0.1)
            acce_lb = torch.clip(
                torch.clip(self.vbound[0] - v, max=self.acce_bound[1]),
                min=self.acce_bound[0],
            )
            acce_ub = torch.clip(
                torch.clip(self.vbound[1] - v, min=self.acce_bound[0]),
                max=self.acce_bound[1],
            )
            lb = torch.cat((acce_lb, -yawbound), dim=-1)
            ub = torch.cat((acce_ub, yawbound), dim=-1)
            return lb, ub

        else:
            raise NotImplementedError

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

    @staticmethod
    def state2yaw(x):
        return x[..., 3:]
    
    @staticmethod
    def state2vel(x):
        return x[..., 2:3]
    
    @staticmethod
    def combine_to_state(xy,vel,yaw):
        if isinstance(xy,torch.Tensor):
            return torch.cat((xy,vel,yaw),-1)
        elif isinstance(xy,np.ndarray):
            return np.concatenate((xy,vel,yaw),-1)

    @staticmethod
    def calculate_vel(pos, yaw, dt, mask):
        if isinstance(pos, torch.Tensor):
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * torch.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * torch.sin(
                yaw[..., 1:, :]
            )
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
            vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * np.cos(
                yaw[..., 1:, :]
            ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * np.sin(yaw[..., 1:, :])
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
    def inverse_dyn(x,xp,dt):
        return (xp[...,2:]-x[...,2:])/dt
    
    @staticmethod
    def get_state(pos,yaw,dt,mask):
        vel = Unicycle.calculate_vel(pos, yaw, dt, mask)
        if isinstance(vel,np.ndarray):
            return np.concatenate((pos,vel,yaw),-1)
        elif isinstance(vel,torch.Tensor):
            return torch.cat((pos,vel,yaw),-1)

    def forward_dynamics(self,
                         initial_states: torch.Tensor,
                         actions: torch.Tensor,
                         step_time: float,
                         mode="chain",
                         bound=True,
                        ):
    
        """
        Integrate the state forward with initial state x0, action u
        Args:
            initial_states (Torch.tensor): state tensor of size [B, (A), 4]
            actions (Torch.tensor): action tensor of size [B, (A), T, 2]
            step_time (float): delta time between steps
        Returns:
            state tensor of size [B, (A), T, 4]
        """
        if mode=="chain":
            return super(Unicycle,self).forward_dynamics(initial_states,actions,step_time,bound=bound)
        else:
            assert mode in ["parallel","partial_parallel"]
            with torch.no_grad():
                num_steps = actions.shape[-2]
                b= initial_states.shape[0]
                device = initial_states.device

                mat = torch.ones(num_steps+1, num_steps+1, device=device)
                mat = torch.tril(mat)
                mat = mat.repeat(b, 1, 1)
                
                mat2 = torch.ones(num_steps, num_steps+1, device=device)
                mat2_h = torch.tril(mat2, diagonal=1)
                mat2_l = torch.tril(mat2, diagonal=-1)
                mat2 = torch.logical_xor(mat2_h, mat2_l).float()*0.5
                mat2 = mat2.repeat(b, 1, 1)
                if initial_states.ndim==3:
                    mat = mat.unsqueeze(1)
                    mat2 = mat2.unsqueeze(1)

            acc = actions[..., :1]
            yaw = actions[..., 1:]
            
            acc_clipped = soft_sat(acc, self.acce_bound[0], self.acce_bound[1])
            
            if mode == 'parallel':

                acc_paded = torch.cat((initial_states[..., -2:-1].unsqueeze(-2), acc_clipped*step_time), dim=-2)
                
                v_raw = torch.matmul(mat, acc_paded)
                v_clipped = soft_sat(v_raw, self.vbound[0], self.vbound[1])
            else:
                v_clipped = [initial_states[..., 2:3]] + [None] * num_steps
                for t in range(num_steps):
                    vt = v_clipped[t]
                    acc_clipped_t = soft_sat(acc_clipped[:, t], self.vbound[0] - vt, self.vbound[1] - vt)
                    v_clipped[t+1] = vt + acc_clipped_t * step_time
                v_clipped = torch.stack(v_clipped, dim=-2)
                
            v_avg = torch.matmul(mat2, v_clipped)
            
            v = v_clipped[..., 1:, :]

            with torch.no_grad():
                v_earlier = v_clipped[..., :-1, :]
                yawbound = torch.minimum(
                    self.max_steer * torch.abs(v_earlier),
                    self.max_yawvel / torch.clip(torch.abs(v_earlier), min=0.1),
                )
                yawbound_clipped = torch.clip(yawbound, min=0.1)
            
            yaw_clipped = soft_sat(yaw, -yawbound_clipped, yawbound_clipped)

            yawvel_paded = torch.cat((initial_states[..., -1:].unsqueeze(-2), yaw_clipped*step_time), dim=-2)
            yaw_full = torch.matmul(mat, yawvel_paded)
            yaw = yaw_full[..., 1:, :]

            # print('before clip', torch.cat((acc[0], yawvel[0]), dim=-1))
            # print('after clip', torch.cat((acc_clipped[0], yawvel_clipped[0]), dim=-1))

            yaw_earlier = yaw_full[..., :-1, :]
            vx = v_avg * torch.cos(yaw_earlier)
            vy = v_avg * torch.sin(yaw_earlier)
            v_all = torch.cat((vx, vy), dim=-1)

            # print('initial_states[0, -2:]', initial_states[0, -2:])
            # print('vx[0, :5]', vx[0, :5])
            
            v_all_paded = torch.cat((initial_states[..., :2].unsqueeze(-2), v_all*step_time), dim=-2)
            x_and_y = torch.matmul(mat, v_all_paded)
            x_and_y = x_and_y[..., 1:, :]

            x_all = torch.cat((x_and_y, v, yaw), dim=-1)
            return x_all, x_and_y, yaw
    def propagate_and_linearize(self,x0,u,dt):
        xp,_,_ = self.forward_dynamics(x0,u,dt,mode="chain")
        xl = torch.cat([x0.unsqueeze(1),xp[:,:-1]],1)
        A,B = jacobian(lambda x,u: self.step(x,u,dt),(xl,u))
        A = A.diagonal(dim1=0,dim2=3).diagonal(dim1=0,dim2=2).permute(2,3,0,1)
        B = B.diagonal(dim1=0,dim2=3).diagonal(dim1=0,dim2=2).permute(2,3,0,1)
        C = xp - (A@xl.unsqueeze(-1)+B@u.unsqueeze(-1)).squeeze(-1)
        return xp,A,B,C




def test():
    model=Unicycle()
    x0 = torch.tensor([[1,2,5,0.1]])
    u = torch.tensor([[0.2,0.05]]).unsqueeze(1).repeat_interleave(10,1)
    xp,A,B,C = model.propagate_and_linearize(x0,u,dt=0.1)

    abc



if __name__=="__main__":
    test()
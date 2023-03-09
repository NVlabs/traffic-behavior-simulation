import torch
import math

from tbsim.dynamics.base import Dynamics, DynType


def bicycle_model(state, acc, ddh, vehicle_length, dt, max_hdot=math.pi * 2.0, max_s=50.0):
    """
    Simple differentiable bicycle model that does not allow reverse
    Args:
        state (torch.Tensor): a batch of current kinematic state [B, ..., 5] (x, y, yaw, speed, hdot)
        acc (torch.Tensor): a batch of acceleration profile [B, ...] (acc)
        ddh (torch.Tensor): a batch of heading acceleration profile [B, ...] (heading)
        vehicle_length (torch.Tensor): a batch of vehicle length [B, ...] (length)
        dt (float): time between steps
        max_hdot (float): maximum change of heading (rad/s)
        max_s (float): maximum speed (m/s)

    Returns:
        New kinematic state (torch.Tensor)
    """
    # state: (x, y, h, speed, hdot)
    assert state.shape[-1] == 5
    newhdot = (state[..., 4] + ddh * dt).clamp(-max_hdot, max_hdot)
    newh = state[..., 2] + dt * state[..., 3].abs() / vehicle_length * newhdot
    news = (state[..., 3] + acc * dt).clamp(0.0, max_s)  # no reverse
    newy = state[..., 1] + news * newh.sin() * dt
    newx = state[..., 0] + news * newh.cos() * dt

    newstate = torch.empty_like(state)
    newstate[..., 0] = newx
    newstate[..., 1] = newy
    newstate[..., 2] = newh
    newstate[..., 3] = news
    newstate[..., 4] = newhdot

    return newstate


class Bicycle(Dynamics):

    def __init__(
            self,
            acc_bound=(-10, 8),
            ddh_bound=(-math.pi * 2.0, math.pi * 2.0),
            max_speed=50.0,
            max_hdot=math.pi * 2.0
    ):
        """
        A simple bicycle dynamics model
        Args:
            acc_bound (tuple): acceleration bound (m/s^2)
            ddh_bound (tuple): angular acceleration bound (rad/s^2)
            max_speed (float): maximum speed, must be positive
            max_hdot (float): maximum turning speed, must be positive
        """
        super(Bicycle, self).__init__(name="bicycle")
        self.xdim = 6
        self.udim = 2
        assert max_speed >= 0
        assert max_hdot >= 0
        self.acc_bound = acc_bound
        self.ddh_bound = ddh_bound
        self.max_speed = max_speed
        self.max_hdot = max_hdot

    def get_normalized_controls(self, u):
        u = torch.sigmoid(u)  # normalize to [0, 1]
        acc = self.acc_bound[0] + (self.acc_bound[1] - self.acc_bound[0]) * u[..., 0]
        ddh = self.ddh_bound[0] + (self.ddh_bound[1] - self.ddh_bound[0]) * u[..., 1]
        return acc, ddh

    def get_clipped_controls(self, u):
        acc = torch.clip(u[..., 0], self.acc_bound[0], self.acc_bound[1])
        ddh = torch.clip(u[..., 1], self.ddh_bound[0], self.ddh_bound[1])
        return acc, ddh

    def step(self, x, u, dt, normalize=True):
        """
        Take a step with the dynamics model
        Args:
            x (torch.Tensor): current state [B, ..., 6] (x, y, h, speed, dh, veh_length)
            u (torch.Tensor): (un-normalized) actions [B, ..., 2] (acc, ddh)
            dt (float): time between steps
            normalize (bool): whether to normalize the actions

        Returns:
            next_x (torch.Tensor): next state after taking the action
        """
        assert x.shape[-1] == self.xdim
        assert u.shape[:-1] == x.shape[:-1]
        assert u.shape[-1] == self.udim
        if normalize:
            acc, ddh = self.get_normalized_controls(u)
        else:
            acc, ddh = self.get_clipped_controls(u)
        next_x = x.clone()  # keep the extent the same
        next_x[..., :5] = bicycle_model(
            state=x[..., :5],
            acc=acc,
            ddh=ddh,
            vehicle_length=x[..., 5],
            dt=dt,
            max_hdot=self.max_hdot,
            max_s=self.max_speed
        )
        return next_x
    @staticmethod
    def calculate_vel(pos, yaw, dt, mask):

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

        return vel

    def type(self):
        return DynType.BICYCLE

    def state2pos(self, x):
        return x[..., :2]

    def state2yaw(self, x):
        return x[..., 2:3]

    def __call__(self, x, u):
        pass

    def ubound(self, x):
        pass

    def name(self):
        return self._name

from tbsim.dynamics.base import DynType, Dynamics
import torch
import numpy as np
from copy import deepcopy


class SingleIntegrator(Dynamics):
    def __init__(self, name, vbound=[1.5,1.5]):
        self._name = name
        self._type = DynType.SI
        self.xdim = vbound.shape[0]
        self.udim = vbound.shape[0]
        self.cyclic_state = list()
        self.vbound = np.array(vbound)

    def __call__(self, x, u):
        assert x.shape[:-1] == u.shape[:, -1]

        return u

    def step(self, x, u, dt, bound=True):
        assert x.shape[:-1] == u.shape[:, -1]
        if bound:
            lb, ub = self.ubound(x)
            if isinstance(x, np.ndarray):
                u = np.clip(u, lb, ub)
            elif isinstance(x, torch.Tensor):
                u = torch.clip(u, min=lb, max=ub)

        return x + u * dt

    def name(self):
        return self._name

    def type(self):
        return self._type

    def ubound(self, x):
        if isinstance(x, np.ndarray):
            lb = np.ones_like(x) * self.vbound[:, 0]
            ub = np.ones_like(x) * self.vbound[:, 1]
            return lb, ub
        elif isinstance(x, torch.Tensor):
            lb = torch.ones_like(x) * torch.from_numpy(self.vbound[:, 0])
            ub = torch.ones_like(x) * torch.from_numpy(self.vbound[:, 1])
            return lb, ub
        else:
            raise NotImplementedError

    @staticmethod
    def state2pos(x):
        return x[..., 0:2]

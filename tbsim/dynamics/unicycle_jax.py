from tbsim.dynamics.base import DynType, Dynamics
from tbsim.utils.math_utils import soft_sat
import jax.numpy as jnp
from jax import jacfwd
import jax
import numpy as np
from copy import deepcopy



class UnicycleJax(Dynamics):
    def __init__(
        self, dt, name = None, max_steer=0.5, max_yawvel=6, acce_bound=[-6, 3], vbound=[-10, 40]
    ):
        self._name = name
        self._type = DynType.UNICYCLE
        self.xdim = 4
        self.udim = 2
        self.dt = dt
        self.cyclic_state = [3]
        self.acce_bound = acce_bound
        self.vbound = vbound
        self.max_steer = max_steer
        self.max_yawvel = max_yawvel
        func = lambda x,u:self.step(x,u,dt)
        jacfun = jacfwd(func,argnums=(0,1))
        self.jacfun = jax.jit(jacfun)
        

    def __call__(self, x, u):
        yaw = self.state2yaw(x)
        vel = self.state2vel(x)
        dxdt = jnp.concatenate(
            (jnp.cos(yaw) * vel,
                jnp.sin(yaw) * vel, u),
            axis=-1,
        )
        return dxdt

    def ubound(self, x):
        raise NotImplementedError
    def step(self, x, u, dt):
        yaw = self.state2yaw(x)
        dxdt = jnp.concatenate(
            (
                jnp.cos(yaw) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                jnp.sin(yaw) * (x[..., 2:3] + u[..., 0:1] * dt * 0.5),
                u,
            ),
            axis=-1,
        )
        return x + dxdt * dt

    def name(self):
        return self._name

    def type(self):
        return self._type

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
        return jnp.concatenate((xy,vel,yaw),-1)

    @staticmethod
    def calculate_vel(pos, yaw, dt, mask):
        vel = (pos[..., 1:, 0:1] - pos[..., :-1, 0:1]) / dt * jnp.cos(
            yaw[..., 1:, :]
        ) + (pos[..., 1:, 1:2] - pos[..., :-1, 1:2]) / dt * jnp.sin(
            yaw[..., 1:, :]
        )
        # right finite difference velocity
        vel_r = jnp.concatenate((vel[..., 0:1, :], vel), axis=-2)
        # left finite difference velocity
        vel_l = jnp.concatenate((vel, vel[..., -1:, :]), axis=-2)
        mask_r = jnp.roll(mask, 1, dims=-1)
        mask_r[..., 0] = False
        mask_r = mask_r & mask

        mask_l = jnp.roll(mask, -1, dims=-1)
        mask_l[..., -1] = False
        mask_l = mask_l & mask
        vel = (
            (mask_l & mask_r)[...,np.newaxis] * (vel_r + vel_l) / 2
            + (mask_l & (~mask_r))[...,np.newaxis] * vel_l
            + (mask_r & (~mask_l))[...,np.newaxis] * vel_r
        )
        
        return vel
    
    def obtain_input_constr(self,x,u):
        v = self.state2vel(x).squeeze(-1)
        acce = u[...,0]
        yawrate = u[...,1]
        vclip = jnp.clip(jnp.abs(v),a_min = 0.1,a_max=None)
        # yawbound = jnp.minimum(
        #     self.max_steer * vclip,
        #     self.max_yawvel / vclip,
        # )
        yawbound1 = self.max_steer * vclip + 1e-2
        highspeed_flag = (v>15)
        yawbound2 = (self.max_yawvel / vclip)*highspeed_flag+(~highspeed_flag)*100
        # acce_lb = jnp.clip(
        #     jnp.clip(self.vbound[0] - v, None, self.acce_bound[1]),
        #     self.acce_bound[0],
        #     None,
        # )
        # acce_ub = jnp.clip(
        #     jnp.clip(self.vbound[1] - v, self.acce_bound[0], None),
        #     None,
        #     self.acce_bound[1],
        # )
        # lb = jnp.concatenate((acce_lb, -yawbound),-1)
        # ub = jnp.concatenate((acce_ub, yawbound),-1)
        return jnp.stack((acce-self.acce_bound[0],
                            self.acce_bound[1]-acce,
                            yawrate+yawbound1,
                            yawbound1-yawrate,
                            yawrate+yawbound2,
                            yawbound2-yawrate),-1)
    @staticmethod
    def inverse_dyn(x,xp,dt):
        return (xp[...,2:]-x[...,2:])/dt
    
    @staticmethod
    def get_state(pos,yaw,dt,mask):
        vel = UnicycleJax.calculate_vel(pos, yaw, dt, mask)
        return UnicycleJax.combine_to_state(pos,vel,yaw)

    def forward_dynamics(self,
                         initial_states: jnp.ndarray,
                         actions: jnp.ndarray,
                         step_time: float,
                        ):
    
        """
        Integrate the state forward with initial state x0, action u
        Args:
            initial_states (jnp.array): state tensor of size [B, (A), 4]
            actions (jnp.array): action tensor of size [B, (A), T, 2]
            step_time (float): delta time between steps
        Returns:
            state tensor of size [B, (A), T, 4]
        """
        num_steps = actions.shape[-2]
        x = [initial_states] + [None] * num_steps
        for t in range(num_steps):
            x[t + 1] = self.step(x[t], actions[..., t, :], step_time)

        x = jnp.stack(x[1:], axis=-2)
        return x, UnicycleJax.state2pos(x),UnicycleJax.state2yaw(x)
    def propagate_and_linearize(self,x0,u):
        xp,_,_ = self.forward_dynamics(x0,u,self.dt)
        xl = jnp.concatenate([x0[:,np.newaxis],xp[:,:-1]],1)

        A,B = self.jacfun(xl,u)
        A = A.diagonal(axis1=0,axis2=3).diagonal(axis1=0,axis2=2).transpose(2,3,0,1)
        B = B.diagonal(axis1=0,axis2=3).diagonal(axis1=0,axis2=2).transpose(2,3,0,1)
        C = xp - (A@xl[...,np.newaxis]+B@u[...,np.newaxis]).squeeze(-1)
        return xp,A,B,C




def test():
    model=UnicycleJax(dt=0.1)
    x0 = jnp.array([[1,2,5,0.1]]).repeat(5,0)
    u = jnp.array([[[0.2,0.05]]]).repeat(10,1).repeat(5,0)
    xp,A,B,C = model.propagate_and_linearize(x0,u)
    jit_fun=jax.jit(model.propagate_and_linearize)
    xp,A,B,C = jit_fun(x0,u)
    constr = model.obtain_input_constr(xp,u)
    ubound_fun = jax.jit(model.obtain_input_constr)
    Jac_fun = jacfwd(ubound_fun,argnums=(0,1))
    Jac_fun_v = jax.vmap(Jac_fun,(0,0),0)
    Jac_fun_v2 = jax.vmap(Jac_fun_v,(0,0),0)
    Jac = Jac_fun_v2(xp,u)
    abc



if __name__=="__main__":
    test()
import numpy as np
import jax.numpy as jnp
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.MPC.homotopy import HomotopyType,HOMOTOPY_THRESHOLD
from jax import jacfwd, jacrev
from functools import partial
from jax import jit
from enum import IntEnum
XYH_INDEX = np.array([0,1,3])
from dataclasses import dataclass
from jax.tree_util import register_pytree_node



class AgentType(IntEnum):
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """

    VEHICLE = 3


@dataclass
class road_object:
    type = None
    t0 = None
    x0 = None
    u0 = None
    up = None
    xp = None
    xi = None
    ui = None
    extent = None
    last_seen = None
    updated = False

    def __init__(self, type, t0, x0, u0, up, xp, ui, xi, extent, last_seen, updated):
        self.type = type
        self.t0 = t0
        self.x0 = x0
        self.u0 = u0
        self.up = up
        self.xp = xp
        self.ui = ui
        self.xi = xi
        self.extent = extent
        self.last_seen = last_seen
        self.updated = updated
    def copy(self):
        return road_object(self.type, self.t0, self.x0, self.u0, self.up, self.xp, self.ui, self.xi, self.extent, self.last_seen, self.updated)

@dataclass
class MPC_util_obj:

    LinearizationFunc = dict()
    InputBoundFun = dict()
    InputBoundJac = dict()
    collfun = dict()
    collJac = dict()
    staticcollJac = dict()
    def __init__(self,LinearizationFunc,InputBoundFun,InputBoundJac,collfun,collJac,staticcollJac):
        self.LinearizationFunc = LinearizationFunc
        self.InputBoundFun = InputBoundFun
        self.InputBoundJac = InputBoundJac
        self.collfun = collfun
        self.collJac = collJac
        self.staticcollJac = staticcollJac

def util_flatten(v):
  """Specifies a flattening recipe.

  Params:
    v: the value of registered type to flatten.
  Returns:
    a pair of an iterable with the children to be flattened recursively,
    and some opaque auxiliary data to pass back to the unflattening recipe.
    The auxiliary data is stored in the treedef for use during unflattening.
    The auxiliary data could be used, e.g., for dictionary keys.
  """
  children = (v.LinearizationFunc,v.InputBoundFun,v.InputBoundJac,v.collfun,v.collJac,v.staticcollJac)
  aux_data = None
  return (children, aux_data)

def util_unflatten(aux_data, children):
  """Specifies an unflattening recipe.

  Params:
    aux_data: the opaque data that was specified during flattening of the
      current treedef.
    children: the unflattened children

  Returns:
    a re-constructed object of the registered type, using the specified
    children and auxiliary data.
  """
  return MPC_util_obj(*children)

# # Global registration
register_pytree_node(
    MPC_util_obj,
    util_flatten,    # tell JAX what are the children nodes
    util_unflatten   # tell JAX how to pack back into a MPC_util_obj
)



def Rectangle_free_region_4(xyh: jnp.ndarray, LW: jnp.ndarray):
    """generate 4 disjoint free spaces around a rectangle

    Args:
        xyh (jnp.ndarray): [B,3]
        LW (jnp.array): [B,2]

    Returns:
        A: [B x 4 x 3 x 2]
        b: [B x 4 x 3]
    """
    x=xyh[:,0]
    y=xyh[:,1]
    h=xyh[:,2]
    L = LW[:,0]
    W = LW[:,1]
    bs = x.shape[0]
    A0 = jnp.array([[0., -1.0], [-1., -1.], [1., -1.]])
    A1 = jnp.array([[1.0, 0.], [1., 1.], [1., -1.]])
    A2 = jnp.array([[0., 1.0], [-1., 1.], [1., 1.]])
    A3 = jnp.array([[-1.0, 0.], [-1., 1.], [-1., -1.]])

    A = (
        jnp.expand_dims(jnp.stack((A0, A1, A2, A3), 0),0).repeat(bs, 0)
    )  # B x 4 x 3 x 2

    b0 = jnp.stack((-W / 2, L / 2 - W / 2, L / 2 - W / 2), -1)
    b1 = jnp.stack((-L / 2, W / 2 - L / 2, W / 2 - L / 2), -1)
    b2 = jnp.stack((-W / 2, L / 2 - W / 2, L / 2 - W / 2), -1)
    b3 = jnp.stack((-L / 2, W / 2 - L / 2, W / 2 - L / 2), -1)

    b = jnp.stack((b0, b1, b2, b3), 1)  # B x 4 x 3

    RotM = jnp.concatenate(
        (
            jnp.expand_dims(jnp.stack((jnp.cos(h), jnp.sin(h)), -1),-2),
            jnp.expand_dims(jnp.stack((-jnp.sin(h), jnp.cos(h)), -1),-2),
        ),
        -2
    )  # b x 2 x 2
    RotM = jnp.expand_dims(RotM,1).repeat(4, 1)  # B x 4 x 2 x 2
    offset = jnp.stack((-x, -y), -1)[:, None].repeat(4, 1)

    A = A @ RotM
    b = b - (A @ offset[...,np.newaxis]).squeeze(-1)
    return A, b

def left_shift(x):
    return jnp.concatenate((x[...,1:],x[...,:1]),-1)

def right_shift(x):
    return jnp.concatenate((x[...,-1:],x[...,:-1]),-1)

def Vehicle_coll_constraint(
    ego_xyh: jnp.array,
    ego_lw: jnp.array,
    obj_xyh: jnp.array,
    obj_lw: jnp.array,
    homotopy: jnp.array,
    active_flag: jnp.array = None,
    ignore_undecided=True,
    enforce_type="poly",
    angle_interval = 5,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_xyh (jnp.array): [B x T x 3]
        ego_lw (jnp.array): [B x 2]
        obj_xyh (jnp.array): [B x N x T x 3]
        obj_lw (jnp.array): [B x N x 2]
        active_flag (jnp.array): [B x N x T x 4] (4 comes from the number of regions for vehicle free space)
        homotopy (jnp.array[bool]): [B x N x 3] (3 comes from the number of homotopy classes)
    """
    bs, Na, T = obj_xyh.shape[:3]
    xe,ye,he = ego_xyh.unbind(-1)
    ho = obj_xyh[...,2]
    # ego_xyh_tiled = ego_xyh.repeat_interleave(Na, 0)
    # ego_lw_tiled = ego_lw.repeat_interleave(Na, 0)
    obj_xyh_tiled = obj_xyh.reshape([-1, 3])
    obj_lw_tiled = obj_lw.repeat_interleave(T,1).reshape(-1,2)
    # A, b = TensorUtils.reshape_axisensions(Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled), 0, 1, (bs,Na,T))
    A, b = Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled)
    A = A.reshape([bs,Na,T,4,3,2])
    b = b.reshape([bs,Na,T,4,3])

    # number of free regions
    M = A.shape[-3]

    cornersX = jnp.kron(ego_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))

    cornersY = jnp.kron(ego_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(bs,4,2)
    corners = GeoUtils.batch_rotate_2D(corners.unsqueeze(1), he.unsqueeze(-1).repeat_interleave(4, axis=-1))+ego_xyh[...,None,:2].repeat_interleave(4,-2) # bxTx4x2
    corner_constr = b.unsqueeze(-1)-A@corners[:,None,:,None].transpose(-1,-2) # b x Na x T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.max(3)[0] # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_xyh[:,None,:,None,:2,None]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    if enforce_type=="poly":
        if active_flag is None:
            current_region = center_constr.min(-1)[0].argmax(-1)
            current_flag = jnp.zeros([bs,Na,T,M],dtype=bool)
            current_flag.scatter_(-1,current_region.unsqueeze(-1),1)
            active_flag = current_flag.clone()

            # for homotopy number 1 (CW), left shift the current flag
            active_flag[...,1:,:] = active_flag[...,1:,:] | (left_shift(current_flag[...,:-1,:]) & homotopy[...,1:2].unsqueeze(-2))
            # for homotopy number 2 (CCW), right shift the current flag
            active_flag[...,1:,:] = active_flag[...,1:,:] | (right_shift(current_flag[...,:-1,:]) & homotopy[...,2:3].unsqueeze(-2))
        

        center_constr.masked_fill_(active_flag.unsqueeze(-1),-10)  # mask out regions that are not active
        center_constr = center_constr.max(2)[0]
    elif enforce_type=="angle":
        center_constr = center_constr.max(2)[0]
        delta_path = ego_xyh[:,None,::angle_interval,:2]-obj_xyh[:,:,::angle_interval,:2]
        angle = GeoUtils.round_2pi(jnp.arctan2(delta_path[...,1],delta_path[...,0]))
        angle_constr = -10.0*jnp.ones_like(angle)
        # for homotopy CCW, angle should be larger than 0
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,2].unsqueeze(-1),angle,-10))
        # for homotopy CW, angle should be less than 0
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,1].unsqueeze(-1),-angle,-10))
        # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,0].unsqueeze(-1),HOMOTOPY_THRESHOLD-angle.cumsum(-1).abs(),-10))

        center_constr = jnp.concatenate((center_constr.reshape(bs,Na,-1),angle_constr),-1)

    
    
    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = jnp.kron(obj_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = jnp.kron(obj_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(bs,Na,4,2)
    RotM_obj = jnp.concatenate(
        (
            jnp.stack((jnp.cos(ho), -jnp.sin(ho)), -1).unsqueeze(-2),
            jnp.stack((jnp.sin(ho), jnp.cos(ho)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x Na x T x 2 x 2
    RotM = jnp.concatenate(
        (
            jnp.stack((jnp.cos(he), jnp.sin(he)), -1).unsqueeze(-2),
            jnp.stack((-jnp.sin(he), jnp.cos(he)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x T x 2 x 2
    corners = (RotM_obj.unsqueeze(3).repeat_interleave(4,3)@corners.unsqueeze(2).repeat_interleave(T,2).unsqueeze(-1)).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,None,:,None]@(corners+obj_xyh[...,None,:2]-ego_xyh[:,None,:,None,:2]).unsqueeze(-1)).squeeze(-1)
    obj_corner_constr = jnp.maximum(jnp.abs(corners[...,0])-ego_lw[:,None,None,None,0],jnp.abs(corners[...,1])-ego_lw[:,None,None,None,1])
    if ignore_undecided:
        corner_constr[undecided] = 10.0
        center_constr[undecided] = 10.0
        obj_corner_constr[undecided] = 10.0
    constr = jnp.concatenate((corner_constr.reshape(bs,-1),center_constr.reshape(bs,-1),obj_corner_constr.reshape(bs,-1)),-1)
    return constr

def pedestrian_coll_constraint_angle(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_R: jnp.array,
    homotopy: jnp.array,
    angle_scale = 1.0,
    offsetR=0.0,
    angle_constraint = True,
):
    """generate collision avoidance constraint for pedestrain objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_R (jnp.array): [1]
    """
    
    T = ego_state.shape[0]
    theta = ego_state[..., 3]
    dx = GeoUtils.batch_rotate_2D(obj_state[..., 0:2] - ego_state[:, 0:2], -theta)
    
    marginxy = jnp.stack((jnp.abs(dx[...,0])-ego_lw[0],jnp.abs(dx[...,1])-ego_lw[1]),-1)
    marginp = marginxy.clip(min=0)
    hypot = jnp.linalg.norm(marginp,axis=-1)
    flag = (marginxy<0).any(-1)
    mask = jnp.ones_like(hypot)*-1e4*flag+jnp.ones_like(hypot)*1e4*~flag
    
    hypot=jnp.minimum(hypot,mask)
    margin = jnp.maximum(marginxy.max(-1),hypot)
    undecided = homotopy.sum(-1)>1

    coll_constr = (margin-obj_R-offsetR)
    
    if angle_constraint:
        delta_path = ego_state[:,:2]-obj_state[:,:2]
        angle = GeoUtils.round_2pi(jnp.arctan2(delta_path[...,1],delta_path[...,0]))
        angle_diff = GeoUtils.round_2pi(angle[1:]-angle[:-1]).sum(-1)
        angle_constr = -10.0
        # for homotopy CCW, angle should be larger than 0

        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[2],angle_diff-HOMOTOPY_THRESHOLD,-10))
        # for homotopy CW, angle should be less than 0
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[1],HOMOTOPY_THRESHOLD-angle_diff,-10))
        # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[0],HOMOTOPY_THRESHOLD-jnp.abs(angle_diff),-10))
        mask = 10.0*undecided+-1e4*~undecided
        angle_constr = jnp.maximum(mask,angle_constr)*angle_scale
        
        return jnp.concatenate((coll_constr.reshape,angle_constr.flatten()),0)
    else:
        return coll_constr.flatten()

def pedestrian_coll_constraint_static(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_R: jnp.array,
    angle_interval = 5,
    offsetR=0.0,
):
    """generate collision avoidance constraint for pedestrain objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_R (jnp.array): [1]
    """
    
    T = ego_state.shape[0]
    theta = ego_state[..., 3]
    dx = GeoUtils.batch_rotate_2D(obj_state[..., 0:2] - ego_state[:, 0:2], -theta)
    
    marginxy = jnp.stack((jnp.abs(dx[...,0])-ego_lw[0],jnp.abs(dx[...,1])-ego_lw[1]),-1)
    marginp = marginxy.clip(min=0)
    hypot = jnp.linalg.norm(marginp,axis=-1)
    flag = (marginxy<0).any(-1)
    mask = jnp.ones_like(hypot)*-1e4*flag+jnp.ones_like(hypot)*1e4*~flag
    
    hypot=jnp.minimum(hypot,mask)
    margin = jnp.maximum(marginxy.max(-1),hypot)


    coll_constr = (margin-obj_R-offsetR)

    return coll_constr.flatten()

def Vehicle_coll_constraint_poly(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_lw: jnp.array,
    homotopy: jnp.array,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_lw (jnp.array): [2]
        homotopy (jnp.array[bool]): [3] (3 comes from the number of homotopy classes)
    """
    T = obj_state.shape[0]
    he = ego_state[...,3]
    ho = obj_state[...,3]
    obj_lw_tiled = obj_lw[np.newaxis,:].repeat(T,0)
    
    M = 4
    A, b = Rectangle_free_region_4(obj_state.take(XYH_INDEX,-1), obj_lw_tiled)


    # number of free regions
    

    # cornersX = jnp.kron(ego_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))

    # cornersY = jnp.kron(ego_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    cornersX = (ego_lw[..., 0] + offsetX)*jnp.array([0.5, 0.5, -0.5, -0.5])
    cornersY = (ego_lw[..., 1] + offsetY)*jnp.array([0.5, -0.5, 0.5, -0.5])
    corners = jnp.stack([cornersX, cornersY], axis=-1)
    corners = GeoUtils.batch_rotate_2D(corners[np.newaxis,:], he[:,np.newaxis].repeat(4, axis=1)).reshape(T,4,2)+ego_state[...,np.newaxis,:2].repeat(4,-2) # Tx4x2
    corner_constr = b[...,np.newaxis]-A@(corners.transpose([0,2,1])[:,np.newaxis]) # T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.min(2).max(1) # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_state[:,np.newaxis,:2,np.newaxis]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    
    min_constr = center_constr.min(-1)

    current_flag = min_constr>=min_constr.max(-1)[...,np.newaxis]
    active_flag = jnp.zeros_like(current_flag)|current_flag

    # for homotopy number 1 (CW), left shift the current flag
    active_flag = active_flag.at[1:].max(left_shift(current_flag[:-1]) & homotopy[np.newaxis,1:2])
    # for homotopy number 2 (CCW), right shift the current flag
    active_flag = active_flag.at[1:].max(right_shift(current_flag[:-1]) & homotopy[np.newaxis,2:3])
    # assume that current_flag[-1] = current_flag[0]
    active_flag = active_flag.at[0].max(active_flag[1])

    mask = jnp.ones_like(center_constr)*1e4*active_flag[...,np.newaxis]+jnp.ones_like(center_constr)*-10*~active_flag[...,np.newaxis]

    center_constr = jnp.minimum(center_constr,mask)  # mask out regions that are not active
    center_constr = center_constr.min(2).max(1)

    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = jnp.kron(obj_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = jnp.kron(obj_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(4,2)
    RotM_obj = jnp.stack(
        (
            jnp.stack((jnp.cos(ho), -jnp.sin(ho)), -1),
            jnp.stack((jnp.sin(ho), jnp.cos(ho)), -1),
        ),
        -2,
    )  # T x 2 x 2
    RotM = jnp.stack(
        (
            jnp.stack((jnp.cos(he), jnp.sin(he)), -1),
            jnp.stack((-jnp.sin(he), jnp.cos(he)), -1),
        ),
        -2,
    )  # T x 2 x 2
    corners = (RotM_obj[:,np.newaxis]@corners[np.newaxis,:,:,np.newaxis]).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,np.newaxis]@(corners+obj_state[:,np.newaxis,:2]-ego_state[:,np.newaxis,:2])[...,np.newaxis]).squeeze(-1)
    obj_corner_constr = jnp.maximum(jnp.abs(corners[...,0])-ego_lw[0]/2,jnp.abs(corners[...,1])-ego_lw[1]/2)
    mask = 10.0*undecided+-1e4*~undecided
    corner_constr = jnp.maximum(corner_constr,mask)
    corner_constr = jnp.maximum(corner_constr,mask)
    obj_corner_constr = jnp.maximum(obj_corner_constr,mask)
    constr = jnp.concatenate((corner_constr.flatten(),center_constr.flatten(),obj_corner_constr.flatten()),-1)
    return constr

def Vehicle_coll_constraint_angle(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_lw: jnp.array,
    homotopy: jnp.array,
    offsetX=0.5,
    offsetY=0.3,
    angle_scale = 0.5,
    temp = 5.0,
    angle_constraint = True,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_lw (jnp.array): [2]
        homotopy (jnp.array[bool]): [3] (3 comes from the number of homotopy classes)
    """
    T = obj_state.shape[0]
    he = ego_state[...,3]
    ho = obj_state[...,3]
    obj_lw_tiled = obj_lw[np.newaxis,:].repeat(T,0)

    A, b = Rectangle_free_region_4(obj_state.take(XYH_INDEX,-1), obj_lw_tiled)


    # number of free regions


    # cornersX = jnp.kron(ego_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))

    # cornersY = jnp.kron(ego_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    cornersX = (ego_lw[..., 0] + offsetX)*jnp.array([0.5, 0.5, -0.5, -0.5])
    cornersY = (ego_lw[..., 1] + offsetY)*jnp.array([0.5, -0.5, 0.5, -0.5])
    corners = jnp.stack([cornersX, cornersY], axis=-1)
    corners = GeoUtils.batch_rotate_2D(corners[np.newaxis,:], he[:,np.newaxis].repeat(4, axis=1)).reshape(T,4,2)+ego_state[...,np.newaxis,:2].repeat(4,-2) # Tx4x2
    corner_constr = b[...,np.newaxis]-A@(corners.transpose([0,2,1])[:,np.newaxis]) # T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.min(2).max(1) # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_state[:,np.newaxis,:2,np.newaxis]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    mask = 10.0*undecided+-1e4*~undecided
    center_constr = center_constr.min(2).max(1)
    
    center_constr = jnp.maximum(mask,center_constr)
    

    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = jnp.kron(obj_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = jnp.kron(obj_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(4,2)
    RotM_obj = jnp.stack(
        (
            jnp.stack((jnp.cos(ho), -jnp.sin(ho)), -1),
            jnp.stack((jnp.sin(ho), jnp.cos(ho)), -1),
        ),
        -2,
    )  # T x 2 x 2
    RotM = jnp.stack(
        (
            jnp.stack((jnp.cos(he), jnp.sin(he)), -1),
            jnp.stack((-jnp.sin(he), jnp.cos(he)), -1),
        ),
        -2,
    )  # T x 2 x 2
    corners = (RotM_obj[:,np.newaxis]@corners[np.newaxis,:,:,np.newaxis]).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,np.newaxis]@(corners+obj_state[:,np.newaxis,:2]-ego_state[:,np.newaxis,:2])[...,np.newaxis]).squeeze(-1)
    obj_corner_constr = jnp.maximum(jnp.abs(corners[...,0])-ego_lw[0]/2,jnp.abs(corners[...,1])-ego_lw[1]/2)
    mask = 10.0*undecided+-1e4*~undecided
    corner_constr = jnp.maximum(corner_constr,mask)
    obj_corner_constr = jnp.maximum(obj_corner_constr,mask)
    if angle_constraint:
        delta_path = ego_state[:,:2]-obj_state[:,:2]
        angle = GeoUtils.round_2pi(jnp.arctan2(delta_path[...,1],delta_path[...,0]))
        angle_diff = GeoUtils.round_2pi(angle[1:]-angle[:-1]).sum(-1)
        angle_constr = -10.0
        # for homotopy CCW, angle should be larger than 0

        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[2],angle_diff-HOMOTOPY_THRESHOLD,-10))
        # for homotopy CW, angle should be less than 0
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[1],HOMOTOPY_THRESHOLD-angle_diff,-10))
        # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
        angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[0],HOMOTOPY_THRESHOLD-jnp.abs(angle_diff),-10))
        angle_constr = jnp.maximum(mask,angle_constr)*angle_scale
        return jnp.concatenate((corner_constr.flatten(),center_constr.flatten(),obj_corner_constr.flatten(),angle_constr.flatten()),-1)
    else:
        return jnp.concatenate((corner_constr.reshape(T,-1),center_constr.reshape(T,-1),obj_corner_constr.reshape(T,-1)),-1).flatten()

def Vehicle_coll_constraint_angle1( 
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_lw: jnp.array,
    homotopy: jnp.array,
    offsetX=0.3,
    offsetY=0.2,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_lw (jnp.array): [2]
        homotopy (jnp.array[bool]): [3] (3 comes from the number of homotopy classes)
    """
    T = obj_state.shape[0]
    he = ego_state[...,3]
    ho = obj_state[...,3]


    cornersX = (ego_lw[..., 0])*jnp.array([0.5, 0.5, -0.5, -0.5])
    cornersY = (ego_lw[..., 1])*jnp.array([0.5, -0.5, 0.5, -0.5])
    corners = jnp.stack([cornersX, cornersY], axis=-1)
    dx = (ego_state[..., 0:2] - obj_state[..., 0:2]).repeat(4, axis=-2)
    delta_x1 = GeoUtils.batch_rotate_2D(corners[None,:], he[:,None]) + dx
    delta_x2 = GeoUtils.batch_rotate_2D(delta_x1, -ho.repeat(4, axis=-1))
    corner_constr = jnp.maximum(
        jnp.abs(delta_x2[..., 0]) - 0.5 * obj_lw[..., 0].repeat(4*T, axis=-1)-offsetX,
        jnp.abs(delta_x2[..., 1]) - 0.5 * obj_lw[..., 1].repeat(4*T, axis=-1)-offsetY,
    )
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    
    delta_path = ego_state[:,:2]-obj_state[:,:2]
    angle = GeoUtils.round_2pi(jnp.arctan2(delta_path[...,1],delta_path[...,0]))
    angle_diff = GeoUtils.round_2pi(angle[1:]-angle[:-1]).sum(-1)
    angle_constr = -10.0
    # for homotopy CCW, angle should be larger than 0

    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[2],angle_diff-HOMOTOPY_THRESHOLD,-10))
    # for homotopy CW, angle should be less than 0
    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[1],HOMOTOPY_THRESHOLD-angle_diff,-10))
    # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[0],HOMOTOPY_THRESHOLD-jnp.abs(angle_diff),-10))
    mask = 10.0*undecided+-1e4*~undecided
    angle_constr = jnp.maximum(mask,angle_constr)
    # center_constr = jnp.concatenate((center_constr,angle_constr[None]),-1)

    
    corner_constr = jnp.maximum(corner_constr,mask)
    constr = jnp.concatenate((corner_constr.flatten(),angle_constr.flatten()),-1)
    # return constr
    return corner_constr.flatten()


def Vehicle_coll_constraint_static(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    obj_state: jnp.array,
    obj_lw: jnp.array,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_state (jnp.array): [T x 4]
        ego_lw (jnp.array): [2]
        obj_state (jnp.array): [T x 4]
        obj_lw (jnp.array): [2]
    """
    T = obj_state.shape[0]
    he = ego_state[...,3]
    ho = obj_state[...,3]
    obj_lw_tiled = obj_lw[np.newaxis,:].repeat(T,0)
    
    M = 4
    A, b = Rectangle_free_region_4(obj_state.take(XYH_INDEX,-1), obj_lw_tiled)


    # number of free regions
    

    # cornersX = jnp.kron(ego_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))

    # cornersY = jnp.kron(ego_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    cornersX = (ego_lw[..., 0] + offsetX)*jnp.array([0.5, 0.5, -0.5, -0.5])
    cornersY = (ego_lw[..., 1] + offsetY)*jnp.array([0.5, -0.5, 0.5, -0.5])
    corners = jnp.stack([cornersX, cornersY], axis=-1)
    corners = GeoUtils.batch_rotate_2D(corners[np.newaxis,:], he[:,np.newaxis].repeat(4, axis=1)).reshape(T,4,2)+ego_state[...,np.newaxis,:2].repeat(4,-2) # Tx4x2
    corner_constr = b[...,np.newaxis]-A@(corners.transpose([0,2,1])[:,np.newaxis]) # T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.min(2).max(1) # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_state[:,np.newaxis,:2,np.newaxis]).squeeze(-1)
    # ignore agents with more than 1 homotopies

    
    center_constr = center_constr.min(2).max(1)

    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = jnp.kron(obj_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = jnp.kron(obj_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(4,2)
    RotM_obj = jnp.stack(
        (
            jnp.stack((jnp.cos(ho), -jnp.sin(ho)), -1),
            jnp.stack((jnp.sin(ho), jnp.cos(ho)), -1),
        ),
        -2,
    )  # T x 2 x 2
    RotM = jnp.stack(
        (
            jnp.stack((jnp.cos(he), jnp.sin(he)), -1),
            jnp.stack((-jnp.sin(he), jnp.cos(he)), -1),
        ),
        -2,
    )  # T x 2 x 2
    corners = (RotM_obj[:,np.newaxis]@corners[np.newaxis,:,:,np.newaxis]).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,np.newaxis]@(corners+obj_state[:,np.newaxis,:2]-ego_state[:,np.newaxis,:2])[...,np.newaxis]).squeeze(-1)
    obj_corner_constr = jnp.maximum(jnp.abs(corners[...,0])-ego_lw[0]/2,jnp.abs(corners[...,1])-ego_lw[1]/2)


    constr = jnp.concatenate((corner_constr.flatten(),center_constr.flatten(),obj_corner_constr.flatten()),-1)
    return constr


def polyline_constr(
    ego_state: jnp.array,
    ego_lw: jnp.array,
    polyline: jnp.array,
    direction: jnp.array,
    extra_margin=1.0,
):
    """generate boundary constraint given polyline boundaries

    Args:
        ego_state (jnp.array): [4]
        ego_lw (jnp.array): [2]
        polyline (jnp.array): [L x 3]
        direction (jnp.array): [1]: 1 to stay on the right of line, -1 to stay on the left of the line
        extra_margin (float, default to 0.0): allowed margin
    """

    L,W = ego_lw[0],ego_lw[1]
    delta_x,delta_y,delta_psi = GeoUtils.batch_proj(ego_state[np.newaxis,[0,1,3]], polyline[np.newaxis,:])
    margin = jnp.maximum(L/2*jnp.abs(jnp.sin(delta_psi.squeeze(-1))),W/2*jnp.abs(jnp.cos(delta_psi.squeeze(-1))))
    idx = jnp.abs(delta_x).argmin(1)
    
    delta_y = jnp.take_along_axis(delta_y,idx[...,np.newaxis],1).squeeze(-1)
    return (delta_y*direction+extra_margin-margin).squeeze(-1)


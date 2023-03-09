from typing import OrderedDict
from collections import defaultdict
from functools import partial
import torch
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax import jit
import scipy
import os
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
from tbsim.policies.base import Policy
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.planning_utils as PlanUtils
import tbsim.utils.lane_utils as LaneUtils
import tbsim.utils.vis_utils as VisUtils
from tbsim.utils.timer import Timers
from torch.nn.utils.rnn import pad_sequence
from tbsim.policies.common import Action, Plan
from Pplan.Sampling.spline_planner import SplinePlanner
from Pplan.Sampling.trajectory_tree import TrajTree
from tbsim.dynamics.unicycle import Unicycle
from tbsim.dynamics.double_integrator import DoubleIntegrator
from tbsim.dynamics.unicycle_jax import UnicycleJax
import tbsim.dynamics as dynamics
from tbsim.policies.MPC.homotopy import (
    HomotopyType,
    HomotopyTree,
    identify_homotopy,
    grouping_homotopy_to_tree,
)
from tbsim.policies.MPC.mpc_utils import (
    Vehicle_coll_constraint_angle,
    Vehicle_coll_constraint_angle1,
    pedestrian_coll_constraint_angle,
    polyline_constr,
    road_object,
    AgentType,
)
from enum import IntEnum

# import osqp
# import cvxpy as cp
# from cvxpygen import cpg
import time
from trajdata import MapAPI, VectorMap
from pathlib import Path
import importlib
import pickle
TRAJ_INDEX = [0, 1, 2, 4]
XYH_INDEX = [0, 1, 3]
sim_dt = 0.1
try:
    import forcespro 
    import get_userid
except:
    print("forces pro not found")
from scipy.linalg import block_diag

@partial(jit,static_argnums=(0,1,2,3,12))
def buildObjIneq(T,countx,countu,nt,obj_x0_nt,obj_xp_nt,obj_lw_nt,homotopy_nt,ego_xp,ego_lw,Ox_indices,Ou_indices,ignore_heading_grad=False):
    xdim = dynamic[nt].xdim
    udim = dynamic[nt].udim
    bs = ego_xp.shape[0]
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    obj_xl = jnp.concatenate([obj_x0_nt[None,:, None].repeat(bs,0), obj_xp_nt[:,:, :-1]], -2)
    # input bound constraint  (JOUx/JOUu: Jacobian of Object U bound on x/u)
    dummy_u = jnp.zeros([*obj_xl.shape[:3], dynamic[nt].udim])
    constr_ub = InputBoundFun[nt+"_b"](obj_xl, dummy_u)
    JOUx, JOUu = InputBoundJac[nt+"_b"](obj_xl, dummy_u)
    
    
    # mask out the first derivative because x0 is not in the decision variable
    
    LUO_nt = constr_ub - (JOUx * obj_xl[:, :, :, None]*one_mask[None, None,:,None,None]).sum(-1)
    LUO_nt = LUO_nt.reshape(bs,-1)

    # collision avoidance constraint  (JCEx/JCOx: Jacobian of Collision on Ego/Object x )
    constr_coll_nt = collfun[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    JCEx, JCOx = collJac[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    if ignore_heading_grad:
        JCEx = JCEx.at[...,3].set(0)
        JCOx = JCOx.at[...,3].set(0)

    LC_nt = constr_coll_nt - (
        JCEx * ego_xp[:, None, None]
        + JCOx * obj_xp_nt[:, :, None]
    ).sum((3, 4))

    LC_nt = LC_nt.reshape(bs, -1)

    JCEx, JCOx = TensorUtils.join_dimensions((JCEx, JCOx), 3, 5)
    GC_nt = jnp.zeros(
        [bs, constr_coll_nt.shape[1] * constr_coll_nt.shape[2], countx]
    )

    GUx_nt = jnp.zeros(
        [
            bs,
            constr_ub.shape[1] * constr_ub.shape[2] * constr_ub.shape[3],
            countx,
        ]
    )
    GUu_nt = jnp.zeros(
        [
            bs,
            constr_ub.shape[1] * constr_ub.shape[2] * constr_ub.shape[3],
            countu,
        ]
    )
    # collsiion constraint entries for ego
    GC_nt = GC_nt.at[..., : xdim * T].set(TensorUtils.join_dimensions(-JCEx, 1, 3))
    # collsiion constraint entries for obj
    J_mat = TensorUtils.block_diag_from_cat_jit(JCOx)
    GC_nt = GC_nt.at[..., Ox_indices].set(-J_mat)

    # Input bound constraint entries
    JOUx_mat = TensorUtils.block_diag_from_cat_jit(JOUx)
    N = JOUx.shape[-2]
    # shift JOUx_mat left by xdim because JOUx is on obj_xl and we need Jacobian on obj_xp
    JOUx_mat = JOUx_mat.at[:,:,:N].set(0)
    JOUx_mat = jnp.roll(JOUx_mat,-xdim,3)
    JOUx_mat = JOUx_mat.at[:,:,:,xdim*(T-1):].set(0)
    JOUx_mmat = TensorUtils.block_diag_from_cat_jit(JOUx_mat)
    JOUu_mat = TensorUtils.block_diag_from_cat_jit(JOUu)
    JOUu_mmat = TensorUtils.block_diag_from_cat_jit(JOUu_mat)
    GUx_nt = GUx_nt.at[:,:, Ox_indices].set(-JOUx_mmat)
    GUu_nt = GUu_nt.at[:,:, Ou_indices].set(-JOUu_mmat)
    return constr_coll_nt,GUx_nt,GUu_nt,LUO_nt,GC_nt,LC_nt

@partial(jit,static_argnums=(0,1,2,10,11))
def buildObjIneq_stage(T,nt,Na,obj_x0_nt,obj_xp_nt,obj_lw_nt,homotopy_nt,ego_xp,ego_lw,agent_idx,ignore_heading_grad=False,sparsity=False):
    xdim = dynamic[nt].xdim
    udim = dynamic[nt].udim
    bs = ego_xp.shape[0]
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    obj_xl = jnp.concatenate([obj_x0_nt[None,:, None].repeat(bs,0), obj_xp_nt[:,:, :-1]], -2)
    # input bound constraint  (JOUx/JOUu: Jacobian of Object U bound on x/u)
    dummy_u = jnp.zeros([*obj_xl.shape[:3], dynamic[nt].udim])
    constr_ub = InputBoundFun[nt+"_b"](obj_xl, dummy_u)
    JOUx, JOUu = InputBoundJac[nt+"_b"](obj_xl, dummy_u)
    if sparsity:
        JOUx = jnp.ones_like(JOUx)
        JOUu = jnp.ones_like(JOUu)
    
    
    # mask out the first derivative because x0 is not in the decision variable
    
    LUO_nt = constr_ub - (JOUx * obj_xl[:, :, :, None]*one_mask[None, None,:,None,None]).sum(-1)


    # collision avoidance constraint  (JCEx/JCOx: Jacobian of Collision on Ego/Object x )
    constr_coll_nt = collfun[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    JCEx, JCOx = collJac[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    if sparsity:
        JCEx = jnp.ones_like(JCEx)
        JCOx = jnp.ones_like(JCOx)
    if ignore_heading_grad:
        JCEx = JCEx.at[...,3].set(0)
        JCOx = JCOx.at[...,3].set(0)

    LC_nt = constr_coll_nt - (
        JCEx * ego_xp[:, None, None]
        + JCOx * obj_xp_nt[:, :, None]
    ).sum((3, 4))
    Na_nt = obj_x0_nt.shape[0]
    LC_nt = LC_nt.reshape(bs, Na_nt, T, -1).transpose(0,2,1,3).reshape(bs,T,-1)
    constr_coll_nt = constr_coll_nt.reshape(bs,Na_nt,T,-1).transpose(0,2,1,3)
    d = constr_coll_nt.shape[-1]
    JCEx = JCEx.reshape(bs, Na_nt, T, -1, T, xdim)
    JCEx = JCEx.diagonal(axis1=2,axis2=4).transpose(0,4,1,2,3)
    JCOx = JCOx.reshape(bs, Na_nt, T, -1, T, xdim)
    JCOx = JCOx.diagonal(axis1=2,axis2=4).transpose(0,4,1,2,3)
    GCO = jnp.zeros([bs,T,Na,d,xdim])
    GCO = GCO.at[:,:,agent_idx].set(-JCOx)
    GCO = TensorUtils.block_diag_from_cat_jit(GCO)
    GCE = -TensorUtils.join_dimensions(JCEx,2,4)
    GC = jnp.concatenate([GCE,GCO],-1)
    
    JOUx = JOUx.transpose(0,2,1,3,4)
    JOUu = JOUu.transpose(0,2,1,3,4)
    d = LUO_nt.shape[-1]
    GOUx = jnp.zeros([bs,T,Na,d,xdim])
    GOUx = GOUx.at[:,:,agent_idx].set(-JOUx)
    GOUx = TensorUtils.block_diag_from_cat_jit(GOUx)
    GOUu = jnp.zeros([bs,T,Na,d,udim])
    GOUu = GOUu.at[:,:,agent_idx].set(-JOUu)
    GOUu = TensorUtils.block_diag_from_cat_jit(GOUu)
    LUO_nt = LUO_nt.transpose(0,2,1,3).reshape(bs,T,-1)
    return constr_coll_nt,GC,LC_nt,GOUx,GOUu,LUO_nt

def ObjIneqSparsity(T,countx,countu,nt,obj_x0_nt,obj_xp_nt,obj_lw_nt,homotopy_nt,ego_xp,ego_lw,Ox_indices,Ou_indices):
    xdim = dynamic[nt].xdim
    udim = dynamic[nt].udim
    bs = ego_xp.shape[0]
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    obj_xl = jnp.concatenate([obj_x0_nt[None,:, None].repeat(bs,0), obj_xp_nt[:,:, :-1]], -2)
    # input bound constraint  (JOUx/JOUu: Jacobian of Object U bound on x/u)
    dummy_u = jnp.zeros([*obj_xl.shape[:3], dynamic[nt].udim])
    constr_ub = InputBoundFun[nt+"_b"](obj_xl, dummy_u)
    JOUx, JOUu = InputBoundJac[nt+"_b"](obj_xl, dummy_u)
    JOUxone = jnp.ones_like(JOUx)
    JOUuone = jnp.ones_like(JOUu)
    

    # collision avoidance constraint  (JCEx/JCOx: Jacobian of Collision on Ego/Object x )
    constr_coll_nt = collfun[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    JCEx, JCOx = collJac[nt](
        ego_xp, ego_lw, obj_xp_nt, obj_lw_nt, homotopy_nt
    )
    JCExone = jnp.ones_like(JCEx)
    JCOxone = jnp.ones_like(JCOx)


    JCExone, JCOxone = TensorUtils.join_dimensions((JCExone, JCOxone), 3, 5)

    GC_nt = jnp.zeros(
        [bs, constr_coll_nt.shape[1] * constr_coll_nt.shape[2], countx]
    )

    GUx_nt = jnp.zeros(
        [
            bs,
            constr_ub.shape[1] * constr_ub.shape[2] * constr_ub.shape[3],
            countx,
        ]
    )
    GUu_nt = jnp.zeros(
        [
            bs,
            constr_ub.shape[1] * constr_ub.shape[2] * constr_ub.shape[3],
            countu,
        ]
    )
    # collsiion constraint entries for ego
    GC_nt = GC_nt.at[..., : xdim * T].set(TensorUtils.join_dimensions(JCExone, 1, 3))
    # collsiion constraint entries for obj
    J_matone = TensorUtils.block_diag_from_cat_jit(JCOxone)
    GC_nt = GC_nt.at[..., Ox_indices].set(J_matone)

    # Input bound constraint entries
    JOUx_matone = TensorUtils.block_diag_from_cat_jit(JOUxone)
    N = JOUx.shape[-2]
    # shift JOUx_mat left by xdim because JOUx is on obj_xl and we need Jacobian on obj_xp
    JOUx_matone = JOUx_matone.at[:,:,:N].set(0)
    JOUx_matone = jnp.roll(JOUx_matone,-N,3)
    JOUx_matone = JOUx_matone.at[:,:,:,xdim*(T-1):].set(0)
    JOUx_mmatone = TensorUtils.block_diag_from_cat_jit(JOUx_matone)
    JOUu_matone = TensorUtils.block_diag_from_cat_jit(JOUuone)
    JOUu_mmatone = TensorUtils.block_diag_from_cat_jit(JOUu_matone)
    GUx_nt = GUx_nt.at[:,:, Ox_indices].set(JOUx_mmatone)
    GUu_nt = GUu_nt.at[:,:, Ou_indices].set(JOUu_mmatone)
    return GUx_nt,GUu_nt,GC_nt

@partial(jit,static_argnums=(0,6))
def buildstaticEgoIneq(nt,obj_xp_nt,obj_lw_nt,ego_xp,ego_lw,homotopy,ignore_heading_grad):

    bs = ego_xp.shape[0]
    # collision avoidance constraint  (JCEx/JCOx: Jacobian of Collision on Ego/Object x )
    constr_coll_nt = collfun[nt](ego_xp, ego_lw, obj_xp_nt, obj_lw_nt,homotopy)
    JCEx = staticcollJac[nt](ego_xp, ego_lw, obj_xp_nt, obj_lw_nt,homotopy)

    if ignore_heading_grad:
        JCEx = JCEx.at[..., 3].set(0)

    LC_nt = constr_coll_nt - (JCEx * ego_xp[:, None, None]).sum((3, 4))
    LC_nt = LC_nt.reshape(bs, -1)

    JCEx = TensorUtils.join_dimensions(JCEx, 3, 5)

    # collsiion constraint entries for ego
    GC_nt = TensorUtils.join_dimensions(-JCEx, 1, 3)

    return constr_coll_nt,GC_nt,LC_nt

@partial(jit,static_argnums=(0,6,7))
def buildstaticEgoIneq_stage(nt,obj_xp_nt,obj_lw_nt,ego_xp,ego_lw,homotopy,ignore_heading_grad,sparsity=False):
    xdim = 4
    bs, Na_nt, T = obj_xp_nt.shape[:3]
    # collision avoidance constraint  (JCEx/JCOx: Jacobian of Collision on Ego/Object x )
    constr_coll_nt = collfun[nt](ego_xp, ego_lw, obj_xp_nt, obj_lw_nt,homotopy)
    JCEx = staticcollJac[nt](ego_xp, ego_lw, obj_xp_nt, obj_lw_nt,homotopy)
    if sparsity:
        JCEx = jnp.ones_like(JCEx)
    if ignore_heading_grad:
        JCEx = JCEx.at[..., 3].set(0)

    LC_nt = constr_coll_nt - (JCEx * ego_xp[:, None, None]).sum((3, 4))
    LC_nt = LC_nt.reshape(bs, Na_nt, T, -1)
    d = LC_nt.shape[-1]
    LC_nt = LC_nt.transpose(0,2,1,3).reshape(bs,T,-1)
    constr_coll_nt = constr_coll_nt.reshape(bs,Na_nt,T,-1).transpose(0,2,1,3).reshape(bs,T,-1)
    GC_nt = -JCEx.reshape(bs,Na_nt,T,d,T,-1).diagonal(axis1=2,axis2=4).transpose(0,4,1,2,3).reshape(bs,T,-1,xdim)

    return constr_coll_nt,GC_nt,LC_nt



@partial(jit,static_argnums=(0,1,6))
# @jit
def buildEgoIneq(countx,countu,ego_x0,ego_xp,ego_lw,lane_info,ignore_heading_grad=False):
    bs, T = ego_xp.shape[:2]
    xdim = 4
    udim = 2
    # setup input bound constraint (JEUx/JEUu: Jacobian of Ego U bound on x/u)
    dummy_u = jnp.zeros([bs, T, udim])
    ego_xl = jnp.concatenate((ego_x0[:, None], ego_xp[:, :-1]), 1)
    constr = InputBoundFun["VEHICLE"](ego_xl, dummy_u)
    JEUx, JEUu = InputBoundJac["VEHICLE"](ego_xl, dummy_u)
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    # mask out the first derivative because x0 is not in the decision variable
    LUE = constr - (JEUx * jnp.expand_dims(ego_xl, 2)*one_mask[None,:,None,None]).sum(-1)
    
    
    LUE = LUE.reshape(bs, -1)
    GUxE = jnp.zeros([bs, constr.shape[1] * constr.shape[2], countx])
    GUuE = jnp.zeros([bs, constr.shape[1] * constr.shape[2], countu])
    JEUx_mat = TensorUtils.block_diag_from_cat_jit(JEUx)
    # shift JEUx_mat left by 1 because JEUx is on ego_xl and we need Jacobian on ego_xp
    N = JEUx.shape[2]
    JEUx_mat = JEUx_mat.at[:,:N].set(0)
    JEUx_mat = jnp.roll(JEUx_mat,-xdim,2)
    JEUx_mat = JEUx_mat.at[:,:,xdim*(T-1):].set(0)
    JEUu_mat = TensorUtils.block_diag_from_cat_jit(JEUu)
    GUxE = GUxE.at[..., : xdim * T].set(-JEUx_mat)
    GUuE = GUuE.at[..., : udim * T].set(-JEUu_mat)
    if lane_info is not None:
        polyline = list()
        direction = list()
        if lane_info["leftbdry"] is not None:
            polyline.append(lane_info["leftbdry"])
            direction.append(-1)
        if lane_info["rightbdry"] is not None:
            polyline.append(lane_info["rightbdry"])
            direction.append(1)
        if len(polyline)>0:
            polyline = jnp.stack(polyline,0)
            direction = jnp.array(direction)
            constr_lane = collfun["LANE"](ego_xp, ego_lw, polyline, direction).reshape(bs,-1)
            Jlane = collJac["LANE"](ego_xp, ego_lw, polyline, direction)
            if ignore_heading_grad:
                Jlane = Jlane.at[...,3].set(0)
            Jlanemat = TensorUtils.block_diag_from_cat_jit(Jlane)
            GL = -Jlanemat
            LL = constr_lane-(Jlanemat@ego_xp.reshape(bs,-1,1)).squeeze(-1)
        else:
            GL = jnp.zeros([T*2,xdim*T])
            LL = jnp.ones(T*2)
            constr_lane = jnp.ones(T*2)
    else:
        GL = jnp.zeros([T*2,xdim*T])
        LL = jnp.ones(T*2)
        constr_lane = jnp.ones(T*2)
        
    return GUxE,GUuE,LUE,GL,LL,constr_lane

@partial(jit,static_argnums=(4,5))
# @jit
def buildEgoIneq_stage(ego_x0,ego_xp,ego_lw,lane_info,ignore_heading_grad=False,sparsity=False):
    bs, T = ego_xp.shape[:2]
    xdim = 4
    udim = 2
    # setup input bound constraint (JEUx/JEUu: Jacobian of Ego U bound on x/u)
    dummy_u = jnp.zeros([bs, T, udim])
    ego_xl = jnp.concatenate((ego_x0[:, None], ego_xp[:, :-1]), 1)
    constr = InputBoundFun["VEHICLE"](ego_xl, dummy_u)
    JEUx, JEUu = InputBoundJac["VEHICLE"](ego_xl, dummy_u)
    if sparsity:
        JEUx = jnp.ones_like(JEUx)
        JEUu = jnp.ones_like(JEUu)
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    # mask out the first derivative because x0 is not in the decision variable
    LUE = constr - (JEUx * jnp.expand_dims(ego_xl, 2)*one_mask[None,:,None,None]).sum(-1)
    

    if lane_info is not None:
        polyline = list()
        direction = list()
        if lane_info["leftbdry"] is not None:
            polyline.append(lane_info["leftbdry"])
            direction.append(-1)
        if lane_info["rightbdry"] is not None:
            polyline.append(lane_info["rightbdry"])
            direction.append(1)
        if len(polyline)>0:
            polyline = jnp.stack(polyline,0)
            direction = jnp.array(direction)
            constr_lane = collfun["LANE"](ego_xp, ego_lw, polyline, direction).reshape(bs,T,-1)
            Jlane = collJac["LANE"](ego_xp, ego_lw, polyline, direction)
            if sparsity:
                Jlane = jnp.ones_like(Jlane)
            if ignore_heading_grad:
                Jlane = Jlane.at[...,3].set(0)

            GL = -Jlane
            LL = constr_lane-((Jlane*ego_xp[:,:,None])).sum(-1)
        else:
            GL = jnp.zeros([bs,T,2,xdim])
            LL = jnp.ones(bs,T,2)
            constr_lane = jnp.ones(bs,T,2)
    else:
        GL = jnp.zeros([bs,T,2,xdim])
        LL = jnp.ones(bs,T,2)
        constr_lane = jnp.ones(bs,T,2)
        
    return -JEUx,-JEUu,LUE,GL,LL,constr_lane

def EgoIneqSparsity(countx,countu,ego_x0,ego_xp,ego_lw,lane_info):
    bs, T = ego_xp.shape[:2]
    xdim = 4
    udim = 2
    # setup input bound constraint (JEUx/JEUu: Jacobian of Ego U bound on x/u)
    dummy_u = jnp.zeros([bs, T, udim])
    ego_xl = jnp.concatenate((ego_x0[:, None], ego_xp[:, :-1]), 1)
    constr = InputBoundFun["VEHICLE"](ego_xl, dummy_u)
    JEUx, JEUu = InputBoundJac["VEHICLE"](ego_xl, dummy_u)
    JEUxone = jnp.ones_like(JEUx)
    JEUuone = jnp.ones_like(JEUu)
    one_mask = jnp.ones(T)
    one_mask = one_mask.at[0].set(0)
    # mask out the first derivative because x0 is not in the decision variable

    GUxE = jnp.zeros([bs, constr.shape[1] * constr.shape[2], countx])
    GUuE = jnp.zeros([bs, constr.shape[1] * constr.shape[2], countu])
    JEUx_matone = TensorUtils.block_diag_from_cat_jit(JEUxone)
    # shift JEUx_mat left by 1 because JEUx is on ego_xl and we need Jacobian on ego_xp
    JEUx_matone = JEUx_matone.at[:,:2*udim].set(0)
    JEUx_matone = jnp.roll(JEUx_matone,-xdim,2)
    JEUx_matone = JEUx_matone.at[:,:,xdim*(T-1):].set(0)
    JEUu_matone = TensorUtils.block_diag_from_cat_jit(JEUuone)
    GUxE = GUxE.at[..., : xdim * T].set(-JEUx_matone)
    GUuE = GUuE.at[..., : udim * T].set(-JEUu_matone)
    if lane_info is not None:
        polyline = list()
        direction = list()
        if lane_info["leftbdry"] is not None:
            polyline.append(lane_info["leftbdry"])
            direction.append(-1)
        if lane_info["rightbdry"] is not None:
            polyline.append(lane_info["rightbdry"])
            direction.append(1)
        if len(polyline)>0:
            polyline = jnp.stack(polyline,0)
            direction = jnp.array(direction)
            constr_lane = collfun["LANE"](ego_xp, ego_lw, polyline, direction).reshape(bs,-1)
            Jlane = collJac["LANE"](ego_xp, ego_lw, polyline, direction)
            Jlaneone = jnp.ones_like(Jlane)
            Jlanematone = TensorUtils.block_diag_from_cat_jit(Jlaneone)
            GL = -Jlanematone

    return GUxE,GUuE,GL

@partial(jit,static_argnums=0)
def combine_matrices_osqp(totaldim,
                        Ge_ego,
                        h_ego,
                        Ge_obj,
                        h_obj,
                        GUxE,
                        GUuE,
                        LUE,
                        GUxO,
                        GUuO,
                        LUO,
                        G_safety,
                        L_safety,
                        ):
    G_ubound_ego = jnp.concatenate((GUxE,GUuE),-1)  
    G_ubound_obj = jnp.concatenate((GUxO,GUuO),-1)
    #TODO: get from cfg

    bs = G_safety.shape[0]
    
    nslack = G_safety.shape[1]
    
    Ge = jnp.concatenate((Ge_ego,Ge_obj[None,:].repeat(bs,0)),1)
    G_ubound = jnp.concatenate((G_ubound_ego,G_ubound_obj[None,:].repeat(bs, 0)),1)
    G = jnp.concatenate((Ge,G_ubound,G_safety,jnp.zeros([bs,nslack,totaldim])),1)
    H_slack = jnp.concatenate((jnp.zeros([G_ubound.shape[1]+Ge.shape[1],nslack]),-jnp.eye(nslack),-jnp.eye(nslack)),0)[None,:].repeat(bs, 0)
    G = jnp.concatenate((G,H_slack),2)
    LB = jnp.concatenate([h_ego,h_obj[None,:].repeat(bs,0),-jnp.inf*jnp.ones([bs,G_ubound.shape[1]+nslack*2])],1)
    UB = jnp.concatenate([h_ego,h_obj[None,:].repeat(bs,0),LUE,LUO[None,:].repeat(bs,0),L_safety,jnp.zeros([bs,nslack])],1)
    return G,LB,UB,nslack

def combine_matrices(totaldim,
                     Ge_ego,
                     h_ego,
                     Ge_obj,
                     h_obj,
                     GUxE,
                     GUuE,
                     LUE,
                     GUxO,
                     GUuO,
                     LUO,
                     G_safety,
                     L_safety,
                     ):
    G_ubound_ego = jnp.concatenate((GUxE,GUuE),-1)  
    G_ubound_obj = jnp.concatenate((GUxO,GUuO),-1)
    #TODO: get from cfg

    bs = G_safety.shape[0]
    
    nslack = G_safety.shape[1]
    
    Ge = jnp.concatenate((Ge_ego,Ge_obj[None,:].repeat(bs,0)),1)
    he = jnp.concatenate((h_ego,h_obj[None,:].repeat(bs,0)),1)
    G_ubound = jnp.concatenate((G_ubound_ego,G_ubound_obj[None,:].repeat(bs, 0)),1)
    G = jnp.concatenate((G_ubound,G_safety,jnp.zeros([bs,nslack,totaldim])),1)
    H_slack = jnp.concatenate((jnp.zeros([G_ubound.shape[1],nslack]),-jnp.eye(nslack),-jnp.eye(nslack)),0)[None,:].repeat(bs, 0)
    G = jnp.concatenate((G,H_slack),2)
    h = jnp.concatenate([LUE,LUO[None,:].repeat(bs,0),L_safety,jnp.zeros([bs,nslack])],1)
    return Ge,he,G,h,nslack
@partial(jit,static_argnums=(0,3))
def buildEgoEq(T,ego_x0,ego_up):
    # TODO: prepare for varying xdim and udim
    xdim = 4
    udim = 2
    ego_xp, A, B, C = LinearizationFunc["VEHICLE"](ego_x0, ego_up)
    bs = ego_x0.shape[0]
    
    L_ego = jnp.zeros([bs, xdim * T])
    L_ego = L_ego.at[:, : xdim * T].set(C.reshape(bs, -1))
    L_ego = L_ego.at[:, :xdim].add((A[:, 0] @ ego_x0[..., None]).squeeze(-1))
    Gx_ego = jnp.eye(xdim * T)[None, :].repeat(bs, 0)
    Gu_ego = jnp.zeros((bs, xdim * T, udim * T))
    Amat = TensorUtils.block_diag_from_cat_jit(A)
    Bmat = TensorUtils.block_diag_from_cat_jit(B)
    Gx_ego = Gx_ego.at[:, xdim : xdim * T, : xdim * (T - 1)].add(-Amat[:, xdim:, xdim:])
    Gu_ego += -Bmat
    return ego_xp,Gx_ego,Gu_ego,L_ego

def EgoEqSparsity(T,ego_x0,ego_up):
    xdim = 4
    udim = 2
    ego_xp, A, B, C = LinearizationFunc["VEHICLE"](ego_x0, ego_up)
    Aone = jnp.ones_like(A)
    Bone = jnp.ones_like(B)
    bs = ego_x0.shape[0]
    Gx_ego = jnp.eye(xdim * T)[None, :].repeat(bs, 0)
    Gu_ego = jnp.zeros((bs, xdim * T, udim * T))
    

    Amatone = TensorUtils.block_diag_from_cat_jit(Aone)
    Bmatone = TensorUtils.block_diag_from_cat_jit(Bone)
    Gx_ego = Gx_ego.at[:, xdim : xdim * T, : xdim * (T - 1)].add(Amatone[:, xdim:, xdim:])
    Gu_ego += Bmatone
    
    return Gx_ego,Gu_ego

@partial(jit,static_argnums=(0,))
def buildObjEq(nt,x0,up,indices,A1_idx,A2_idx,B1_idx,B2_idx,L1_idx,obj_xp,obj_up,Gx_obj,Gu_obj,L_obj):
    xdim=4
    xp, A, B, C = LinearizationFunc[nt](x0, up)
    obj_xp = obj_xp.at[indices].set(xp)
    obj_up = obj_up.at[indices].set(up)
    Amat = TensorUtils.block_diag_from_cat_jit(A)[:, xdim:, xdim:]
    AAmat = TensorUtils.block_diag_from_cat_jit(Amat)
    Bmat = TensorUtils.block_diag_from_cat_jit(B)
    BBmat = TensorUtils.block_diag_from_cat_jit(Bmat)


    Gx_obj = Gx_obj.at[jnp.ix_(A1_idx, A2_idx)].add(-AAmat)
    Gu_obj = Gu_obj.at[jnp.ix_(B1_idx, B2_idx)].add(-BBmat)
    
    
    Ax0 = (A[:, 0] @ x0[..., None]).squeeze(-1)
    L_obj = L_obj.at[B1_idx].set(C.flatten())
    L_obj = L_obj.at[L1_idx].add(Ax0.flatten())
    return obj_xp,obj_up,Gx_obj,Gu_obj,L_obj


buildObjEqBatch = jax.vmap(buildObjEq,in_axes=(None,None,0,None,None,None,None,None,None,0,0,0,0,0),out_axes=(0,0,0,0,0))


def ObjEqSparsity(nt,x0,up,A1_idx,A2_idx,B1_idx,B2_idx,Gx_obj,Gu_obj):
    xdim=4
    xp, A, B, C = LinearizationFunc[nt](x0, up)
    Aone = jnp.ones_like(A)
    Bone = jnp.ones_like(B)
    
    Amatone = TensorUtils.block_diag_from_cat_jit(Aone)[:, xdim:, xdim:]
    AAmatone = TensorUtils.block_diag_from_cat_jit(Amatone)
    Bmatone = TensorUtils.block_diag_from_cat_jit(Bone)
    BBmatone = TensorUtils.block_diag_from_cat_jit(Bmatone)

    

    Gx_obj = Gx_obj.at[jnp.ix_(A1_idx, A2_idx)].add(AAmatone)
    Gu_obj = Gu_obj.at[jnp.ix_(B1_idx, B2_idx)].add(BBmatone)
    
    return Gx_obj,Gu_obj

@partial(jit,static_argnums=(0,1))
def buildXrefCost_Q(T,bs,xref,Q,Qf):
    xdim = 4

    cosh = jnp.cos(xref[...,3])
    sinh = jnp.sin(xref[...,3])
    RotM_seq = jnp.stack((
        jnp.stack((cosh,sinh),-1),jnp.stack((-sinh,cosh),-1)
    ),-2)
    RotM_seq = jnp.kron(jnp.array([[1,0],[0,0]]),RotM_seq) + jnp.kron(jnp.array([[0,0],[0,1]]),jnp.eye(2))[None,None,:]
    RotM = TensorUtils.block_diag_from_cat_jit(RotM_seq)
    Q_ego_rot = RotM.transpose(0,2,1)@(Q[None,:].repeat(bs,0))@RotM
    f = - (Q_ego_rot@xref.reshape(bs,-1,1)).squeeze(-1)
    Qmat = Q_ego_rot

    Qf_ego_rot = RotM_seq[:,-1].transpose(0,2,1)@Qf[None,:]@RotM_seq[:,-1]
    f = f.at[:, xdim * (T - 1) :].add((
        -Qf_ego_rot @ xref[:,-1, :,None]
    ).squeeze(-1))
    Qmat = Qmat.at[:,xdim*(T-1):,xdim*(T-1):].add(Qf_ego_rot)
    Qr = []
    return Qmat, f

@partial(jit,static_argnums=(0,1))
# due to the DDP requirement of cvxpy, we need to build cost with sqrtm(Q)
def buildXrefCost_Qr(T,bs,xref,Qr,Qfr):
    xdim = 4

    cosh = jnp.cos(xref[...,3])
    sinh = jnp.sin(xref[...,3])
    RotM_seq = jnp.stack((
        jnp.stack((cosh,sinh),-1),jnp.stack((-sinh,cosh),-1)
    ),-2)
    RotM_seq = jnp.kron(jnp.array([[1,0],[0,0]]),RotM_seq) + jnp.kron(jnp.array([[0,0],[0,1]]),jnp.eye(2))[None,None,:]
    RotM = TensorUtils.block_diag_from_cat_jit(RotM_seq)
    Qr_ego_rot = (Qr[None,:].repeat(bs,0))@RotM
    f = - (Qr_ego_rot.transpose(0,2,1)@Qr_ego_rot@xref.reshape(bs,-1,1)).squeeze(-1)

    Qfr_ego_rot = Qfr[None,:]@RotM_seq[:,-1]
    f = f.at[:, xdim * (T - 1) :].add((
        -Qfr_ego_rot.transpose(0,2,1)@Qfr_ego_rot @ xref[:,-1, :,None]
    ).squeeze(-1))
    return Qr_ego_rot,Qfr_ego_rot, f


@partial(jit,static_argnums=(0,1,5))
def buildXrefCost_stage(T,bs,xref,Q,Qf,rot = False):
    xdim = 4

    
    Q_seq = Q[None,None,:].repeat(bs,0).repeat(T,1)
    Q_seq = Q_seq.at[:,-1].add(Qf[None,:].repeat(bs,0))
    if rot:
        cosh = jnp.cos(xref[...,3])
        sinh = jnp.sin(xref[...,3])
        RotM_seq = jnp.stack((
            jnp.stack((cosh,sinh),-1),jnp.stack((-sinh,cosh),-1)
        ),-2)
        RotM_seq = jnp.kron(jnp.array([[1,0],[0,0]]),RotM_seq) + jnp.kron(jnp.array([[0,0],[0,1]]),jnp.eye(2))[None,None,:]
        Q_ego_rot = RotM_seq.transpose(0,1,3,2)@Q_seq@RotM_seq
    else:
        Q_ego_rot = Q_seq
    f = - (Q_ego_rot@xref[...,None]).squeeze(-1)

    return Q_ego_rot, f



class SQPMPC(object):
    def __init__(self, cfg, ego_sampler, device, qp_solver_dir = None):
        self.cfg = cfg
        self.dt = cfg.dt
        self.pred_dt = cfg.pred_dt
        self.device = device
        self.ego_sampler = ego_sampler

        self.horizon_sec = cfg.horizon_sec
        self.horizon = int(cfg.horizon_sec / self.dt)
        self.stage = 2
        assert self.horizon % self.stage == 0
        self.node_types = {3: "VEHICLE"}
        dyn = cfg.dynamic["ego"]
        kwargs = dyn.attributes if "attributes" in dyn else {}
        self.ego_dyn = Unicycle(self.dt)
        if "dynamic" not in locals():
            global dynamic
            dynamic = dict()
            for nt, dyn in cfg.dynamic.items():

                kwargs = dyn.attributes if "attributes" in dyn else {}
                if dyn.name == "Unicycle":
                    dynamic[nt] = Unicycle(name=nt, **kwargs)
                elif dyn.name == "DoubleIntegrator":
                    dynamic[nt] = DoubleIntegrator(name=nt, **kwargs)
                
                    
            
        self.objects = OrderedDict()
        self.u0 = np.zeros(self.ego_dyn.udim)
        self.ndx = dict()
        self.ndu = dict()
        #TODO: get from cfg
        self.PrepareJitFun()
        self.SetCostParams(cfg)
        self.num_dynamic_object = cfg.num_dynamic_object
        self.num_static_object = cfg.num_static_object
        if "num_static_mode" in cfg:
            self.num_static_mode = cfg.num_static_mode
        else:
            self.num_static_mode=1
        # initialize a dummy object
        t0 = -1
        xp = np.concatenate([2e2*np.ones([self.horizon,2]),np.zeros([self.horizon,2])],-1)
        x0 = xp[0]
        up = np.zeros([self.horizon,2])
        u0 = up[0]
        extent = np.zeros(3)
        last_seen = -1
        updated=True
        self.dummy_object = road_object("VEHICLE", t0, x0, u0, up, xp, up, xp, extent, last_seen, updated)
        self.num_rounds = cfg.num_rounds
        self.solvername = self.cfg.solver_name+"_T"+str(self.horizon)+"_D"+str(self.num_dynamic_object)+"_S"+ str(self.num_static_object) + "_M"+str(self.num_static_mode)
        if cfg.qp_solver == "FORCESPRO":
            # forcespro
            try:
                self.qp_prob = __import__(self.solvername + "_py")
                with open(self.solvername+"/sparsity.pkl", "rb") as f:
                    self.param_sparsity = pickle.load(f)
                
            except:
                self.qp_prob = None
            # self.qp_prob = None
        else:
            # cvxpy
            if qp_solver_dir is None:
                qp_solver_dir = "tbsim/policies/MPC/qp_solver"
            # verify qp solver
            qp_solver_dir = Path(qp_solver_dir)
            absolute_path = qp_solver_dir.resolve()
            if not absolute_path.is_dir():
                os.mkdir(str(absolute_path))
            module_path = str(absolute_path).split('/')
            counter = 0
            while True:
                if module_path[counter] == 'tbsim':
                    break
                counter += 1
            module_path = '.'.join(module_path[counter:])
            self.module_path = module_path
            
            # if (qp_solver_dir/'problem.pickle').is_file(): 
            #     with open(qp_solver_dir/'problem.pickle', 'rb') as f:
            #         prob = pickle.load(f)
            #     if cfg.code_gen:
            #         solver_dir = qp_solver_dir/'cpg_solver.py'
            #         if not solver_dir.is_file():
            #             cpg.generate_code(prob, code_dir=str(self.qp_solver_dir), solver=cfg.qp_solver)
            #         cpg_solver = importlib.import_module(self.module_path+'.cpg_solver')
            #         prob.register_solve('cpg', cpg_solver.cpg_solve)
            # else:
            #     prob = None
            prob = None
            self.qp_solver_dir = qp_solver_dir
            self.qp_prob = prob
        
    def reset(self):
        self.objects.clear()
        

    def SetCostParams(self, cfg):
        # TODO: get this from cfg
        T = self.horizon
        Qveh = jnp.diag(np.array(cfg.MPCCost.VEHICLE.Q))
        Rveh = jnp.diag(np.array(cfg.MPCCost.VEHICLE.R))
        dRveh = jnp.diag(np.array(cfg.MPCCost.VEHICLE.dR))
        Qego = jnp.diag(np.array(cfg.MPCCost.EGO.Q))
        Rego = jnp.diag(np.array(cfg.MPCCost.EGO.R))
        dRego = jnp.diag(np.array(cfg.MPCCost.EGO.dR))
        # penalty for slacks
        self.Mcoll = cfg.MPCCost.Mcoll
        self.Mlane = cfg.MPCCost.Mlane
        self.Qf_ego = jnp.diag(np.array(cfg.MPCCost.EGO.Qf))*self.horizon/3
        self.Q_ego = jnp.kron(jnp.eye(T),Qego)
        self.Qr_ego = jnp.array(scipy.linalg.sqrtm(np.array(self.Q_ego)))
        self.Qfr_ego = jnp.array(scipy.linalg.sqrtm(np.array(self.Qf_ego)))
        self.qego = Qego

        

        self.R = dict(VEHICLE=Rveh)
        self.dR = dict(VEHICLE=dRveh)
        self.Q = dict(VEHICLE=Qveh)
        
        # Q and R
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        R_ego = jnp.kron(jnp.eye(T), Rego)
        # for computation of dR
        S = jnp.zeros([T - 1, T])
        S = S.at[:, :-1].add(-jnp.eye(T - 1))
        S = S.at[:, 1:].add(jnp.eye(T - 1))
        self.S = S
        Su = jnp.kron(S, jnp.eye(udim))
        dR_ego = Su.T @ jnp.kron(jnp.eye(T - 1), dRego) @ Su

        # add the du penalty on the first u
        dR_ego = dR_ego.at[:udim, :udim].add(dRego)

        self.R_ego = R_ego
        self.rego = Rego
        self.drego = dRego
        self.dR_ego = dR_ego
        
        self.joint_weights = dict(ego = cfg.MPCCost.ego_weight,obj=cfg.MPCCost.obj_weight)

    def PrepareJitFun(self):
        # TODO: formalize the jax model and do it for all dynamic types
        # dynamic equation
        if self.cfg.qp_solver=="FORCESPRO":
            assert self.cfg.angle_constraint==False
            self.stage_wise=True
        else:
            self.stage_wise=False    
        if "LinearizationFunc" not in locals():
            global LinearizationFunc,unicyclejax_model,InputBoundFun,InputBoundJac,collfun,collJac,staticcollJac
            unicyclejax_model = UnicycleJax(self.dt)
            veh_lin = jax.jit(unicyclejax_model.propagate_and_linearize)
            veh_lin_v = jax.vmap(veh_lin, (0, 0), 0)
            LinearizationFunc = dict(
                VEHICLE=veh_lin,
                VEHICLE_batch=veh_lin_v
            )
            # input bound
            ubound_fun = jax.jit(unicyclejax_model.obtain_input_constr)
            fun_v = jax.vmap(ubound_fun, (0, 0), 0)
            fun_v2 = jax.vmap(fun_v, (0, 0), 0)
            fun_v3 = jax.vmap(fun_v2, (0, 0), 0)
            Jac_fun = jax.jacfwd(ubound_fun, argnums=(0, 1))
            Jac_fun_v = jax.vmap(Jac_fun, (0, 0), 0)
            Jac_fun_v2 = jax.vmap(Jac_fun_v, (0, 0), 0)
            Jac_fun_v3 = jax.vmap(Jac_fun_v2, (0, 0), 0)
            InputBoundFun = dict(VEHICLE=fun_v2,VEHICLE_b=fun_v3)
            InputBoundJac = dict(VEHICLE=Jac_fun_v2,VEHICLE_b=Jac_fun_v3)
            # vehicle collision avoidance
  

            veh_coll_fun = lambda es,ee,os,oe,h: Vehicle_coll_constraint_angle(es,ee,os,oe,h,offsetX=self.cfg.offsetX,offsetY=self.cfg.offsetY,angle_scale=self.cfg.angle_scale,temp=self.cfg.temp,angle_constraint=self.cfg.angle_constraint)
            veh_fun = jax.jit(veh_coll_fun)
            veh_Jac = jax.jit(jax.jacfwd(veh_fun, argnums=(0, 2)))
            static_veh_Jac = jax.jit(jax.jacfwd(veh_fun, argnums=0))
            #adding agent dimension
            veh_fun_vo = jax.vmap(veh_fun, (None, None, 0, 0, 0), 0)
            #adding ego mode dimension
            veh_fun_vo_ve = jax.vmap(veh_fun_vo, (0, None, 0, None, 0), 0)
            #adding agent dimension
            veh_Jac_vo = jax.vmap(veh_Jac, (None, None, 0, 0, 0), 0)
            #adding ego mode dimension
            veh_Jac_vo_ve = jax.vmap(veh_Jac_vo, (0, None, 0, None, 0), 0)
            static_veh_Jac_vo = jax.vmap(static_veh_Jac, (None, None, 0, 0, 0), 0)
            static_veh_Jac_vo_ve = jax.vmap(static_veh_Jac_vo, (0, None, 0, None, 0), 0)
            
            dummy_input = (
                jnp.zeros([3, 10, 4]),
                jnp.zeros(2),
                jnp.zeros([3, 5, 10, 4]),
                jnp.zeros([5, 2]),
                jnp.zeros([3, 5, 3], dtype=bool),
            )
            _ = veh_fun_vo_ve(*dummy_input)
            _ = veh_Jac_vo_ve(*dummy_input)
            _ = static_veh_Jac_vo_ve(*dummy_input)


            
            
            # pedestrian collision avoidance
            ped_coll_fun = lambda es,ee,os,oe,h: pedestrian_coll_constraint_angle(es,ee,os,oe,h,angle_scale=self.cfg.angle_scale,angle_constraint=self.cfg.angle_constraint)
            ped_fun = jax.jit(ped_coll_fun)
            ped_Jac = jax.jit(jax.jacfwd(ped_fun, argnums=(0, 2)))
            static_ped_Jac = jax.jit(jax.jacfwd(ped_Jac, argnums=0))
            #adding agent dimension
            ped_fun_vo = jax.vmap(ped_fun, (None, None, 0, 0, 0), 0)
            #adding ego mode dimension
            ped_fun_vo_ve = jax.vmap(ped_fun_vo, (0, None, 0, None, 0), 0)
            #adding agent dimension
            ped_Jac_vo = jax.vmap(ped_Jac, (None, None, 0, 0, 0), 0)
            #adding batch dimension
            ped_Jac_vo_ve = jax.vmap(ped_Jac_vo, (0, None, 0, None, 0), 0)
            #adding agent dimension
            static_ped_Jac_vo = jax.vmap(static_ped_Jac, (None, None, 0, 0, 0), 0)
            #adding ego mode dimension
            static_ped_Jac_vo_ve = jax.vmap(static_ped_Jac_vo, (0, None, 0, None, 0), 0)
            dummy_input = (
                jnp.zeros([3, 10, 4]),
                jnp.zeros(2),
                jnp.zeros([3, 5, 10, 4]),
                jnp.zeros([5, 1]),
                jnp.zeros([3, 5, 3], dtype=bool),
            )
            _ = ped_fun_vo_ve(*dummy_input)
            _ = ped_Jac_vo_ve(*dummy_input)
            _ = static_ped_Jac_vo_ve(*dummy_input)
            

            # lane boundary constraint
            func_jit = jax.jit(polyline_constr)
            func_v = jax.vmap(func_jit,(None,None,0,0),0) # adding multiple lanes
            func_vv = jax.vmap(func_v,(0,None,None,None),0) # adding time dimension
            func_vvv = jax.vmap(func_vv,(0,None,None,None),0) # adding batch dimension
            Jac_jit = jax.jit(jax.jacfwd(polyline_constr,argnums=0))
            Jac_v = jax.vmap(Jac_jit,(None,None,0,0),0) # adding multiple lanes
            Jac_vv = jax.vmap(Jac_v,(0,None,None,None),0) # adding time dimension
            Jac_vvv = jax.vmap(Jac_vv,(0,None,None,None),0) # adding batch dimension

            dummy_input = (
                jnp.zeros([3, 10, 4]),
                jnp.zeros(2),
                jnp.zeros([5, 10, 3]),
                jnp.zeros([5]),
            )
            _ = func_vvv(*dummy_input)
            _ = Jac_vvv(*dummy_input)

            collfun = dict(VEHICLE=veh_fun_vo_ve, PEDESTRIAN=ped_fun_vo_ve,LANE=func_vvv)
            collJac = dict(VEHICLE=veh_Jac_vo_ve, PEDESTRIAN=ped_Jac_vo_ve,LANE=Jac_vvv)
            staticcollJac = dict(VEHICLE=static_veh_Jac_vo_ve, PEDESTRIAN=static_ped_Jac_vo_ve)




    def update_road_objects(
        self, time_stamp, x0, u0, pred_x, pred_u, track_ids, obj_type, extents
    ):
        x0, u0, pred_x, pred_u, track_ids, obj_type, extents = TensorUtils.to_numpy(
            (x0, u0, pred_x, pred_u, track_ids, obj_type, extents)
        )
        if track_ids is not None:
            track_ids = track_ids.tolist()
        else:
            track_ids = np.arange(1,x0.shape[0]+1).tolist()
        obj_type = obj_type.tolist()

        for i, (track_id, type, extent) in enumerate(zip(track_ids, obj_type, extents)):
            if track_id < 0:
                continue
            #TODO: implement the update logic
            # if (
            #     track_id not in self.objects
            #     or time_stamp - self.objects[track_id].last_seen > self.cfg.delta_t_max
            # ):
            if type>0:
                # if object is not seen before or is outdated, create new object
                self.objects[track_id] = road_object(
                    type=self.node_types[type],
                    t0=time_stamp,
                    x0=x0[i],
                    u0=u0[0,i],
                    up=pred_u[:,i],
                    xp=pred_x[:,i],
                    ui = None,
                    xi = None,
                    last_seen=time_stamp,
                    extent=extent,
                    updated=False,
                )
            # else:
            #     raise NotImplementedError
            #     # update object
            #     obj_i = self.objects[track_id]
            #     delta_t = time_stamp - obj_i.last_seen
            #     x0_i = obj_i.xp[delta_t - 1]
            #     up = torch.cat(obj_i.up[delta_t:], torch.zeros([delta_t, 2]), 0)
            #     xp_extend = Unicycle.forward_dynamics()

    @staticmethod
    def sampling_cost_fun(
        weights,
        ego_trajs,
        pred,
        ego_extents,
        agent_extents,
        agent_types,
        ego_goal = None,
        lane_info=None,
        col_funcs=None,
    ):
        Ne = ego_trajs.shape[0]
        numMode = pred.shape[0]
        col_loss = PlanUtils.get_collision_loss(
            ego_trajs.unsqueeze(1).repeat_interleave(numMode, 1),
            pred.unsqueeze(0).repeat_interleave(Ne, 0),
            ego_extents[None, ..., :2].repeat_interleave(Ne, 0),
            agent_extents[None, ..., :2].repeat_interleave(Ne,0),
            agent_types[None,:].repeat_interleave(Ne,0),
            col_funcs,
        )

        if lane_info is not None:
            polyline = list()
            direction = list()
            if lane_info["leftbdry"] is not None:
                polyline.append(lane_info["leftbdry"])
                direction.append(-1)
            if lane_info["rightbdry"] is not None:
                polyline.append(lane_info["rightbdry"])
                direction.append(1)
            if len(polyline)>0:
                polyline = jnp.stack(polyline,0)
                direction = jnp.array(direction)
                constr_lane = collfun["LANE"](TensorUtils.to_jax(ego_trajs), TensorUtils.to_jax(ego_extents), polyline, direction[...,None])
                lane_loss = TensorUtils.to_torch(np.array((constr_lane.min(2).min(1)).clip(max=0.0)),device=ego_trajs.device)
                lane_loss = lane_loss[:,None]
                
        else:
            lane_loss = 0


        if ego_goal is not None:
            ego_goal  = TensorUtils.to_torch(ego_goal,device=ego_trajs.device)
            goal_loss = torch.norm(ego_goal[None,:2]-ego_trajs[:,-1,:2],dim=-1,keepdim=True).clip(max=10.0)
        else:
            goal_loss = 0.0
        total_loss = (
            weights["collision_weight"] * col_loss
            + weights["lane_weight"] * lane_loss
            + weights["goal_reaching"] * goal_loss
        )
        return total_loss

    def select_homotopy(self, ego_trajs, pred, sample_cost):
        Ne = ego_trajs.shape[0]
        if pred.nelement()==0:
            # no agents around
            homotopy_unique =torch.zeros((Ne,0,3),device=ego_trajs.device)
            obj_idx=torch.zeros(Ne,device=ego_trajs.device,dtype=torch.int64)
            sample_cost = sample_cost[:,0]
            return homotopy_unique, ego_trajs, obj_idx, sample_cost
        numMode = pred.shape[0]
        _, homotopy = identify_homotopy(
            ego_trajs[..., :2], pred.unsqueeze(0).repeat_interleave(Ne, 0)[..., :2]
        )
        homotopy = TensorUtils.join_dimensions(homotopy,0,2)
        homotopy_unique = homotopy.unique(dim=0)
        hc_list = (homotopy.unsqueeze(0) == homotopy_unique.unsqueeze(1)).all(-1)
        optimal_idx = torch.argmin(torch.where(hc_list, sample_cost.flatten(), np.inf), dim=1)
        ego_idx = optimal_idx // numMode
        obj_idx = optimal_idx % numMode
        
        return homotopy_unique, ego_trajs[ego_idx], obj_idx, sample_cost.flatten()[optimal_idx]

    def setup_mpc_instances(self, ego_x0, ego_extent, ego_xp, ego_up, dyn_homotopy,static_homotopy, dynamic_objects,static_objects,obj_modes,lane_info,xref=None):
        # arange decision variables
        # ego_x0,ego_extent,ego_xp,ego_up,homotopy = TensorUtils.to_numpy((ego_x0,ego_extent,ego_xp,ego_up,homotopy))
        
        ego_x0 = ego_x0[None,:].repeat(ego_xp.shape[0],0)

        countx = 0
        countu = 0
        T = self.horizon
        ndx = {"ego": 0}
        ndu = {"ego": 0}
        countx = dynamic["VEHICLE"].xdim * T
        countu = dynamic["VEHICLE"].udim * T

        for track_id, obj in dynamic_objects.items():
            ndx[track_id] = countx
            ndu[track_id] = countu
            countx += dynamic[obj.type].xdim * T
            countu += dynamic[obj.type].udim * T
        self.ndx = ndx
        self.ndu = ndu
        self.countx = countx
        self.countu = countu
        
        if self.cfg.qp_solver=="FORCESPRO":
            start_time = time.time()
            ego_xp,obj_xp,A,B,C = self.buildEqConstr_stage(
                ego_x0, ego_up, dynamic_objects, obj_modes
            )
            runtime = time.time()-start_time   
            # print(f"build equality constraint takes {runtime}s")

            start_time = time.time()
            (
                GU,
                LU,
                GC,
                LC,
                constr_coll,
                GCS,
                LCS,
                static_constr_coll,
                GL,
                LL,
                constr_lane,
            ) = self.buildIneqConstr_stage(
                ego_x0, ego_xp, obj_xp, ego_extent, dynamic_objects, static_objects, dyn_homotopy,static_homotopy, obj_modes, lane_info=lane_info
            )
            runtime = time.time()-start_time   
            # print(f"build inequality constraint takes {runtime}s")
            start_time = time.time()
            R, fu, Q, fx, dR = self.buildCost_stage(
                self.u0, ego_xp, dynamic_objects, obj_modes, xref,  self.joint_weights
            )
            runtime = time.time()-start_time   
            # print(f"build cost takes {runtime}s")
            if self.qp_prob is None:
                (
                    sGU,
                    _,
                    sGC,
                    _,
                    _,
                    sGCS,
                    _,
                    _,
                    sGL,
                    _,
                    _,
                ) = self.buildIneqConstr_stage(
                    ego_x0, ego_xp, obj_xp, ego_extent, dynamic_objects, static_objects, dyn_homotopy,static_homotopy, obj_modes, lane_info=lane_info, sparsity=True
                )
                self.build_qp_solver_forcespro(ego_xp,obj_xp,A,B,C,sGU,LU,sGC,LC,sGCS,LCS,sGL,LL,R,fu,Q,fx,dR)
            ego_xp,obj_xp,A,B,C,GU,LU,GC,LC,GCS,LCS,GL,LL,R,fu,Q,fx,dR = TensorUtils.to_numpy((ego_xp,obj_xp,A,B,C,GU,LU,GC,LC,GCS,LCS,GL,LL,R,fu,Q,fx,dR))
            return ego_xp,obj_xp,A,B,C,GU,LU,GC,LC,GCS,LCS,GL,LL,R,fu,Q,fx,dR
        else:
            start_time = time.time()
            Ge_ego, h_ego, ego_xp, Ge_obj, h_obj, obj_xp,obj_up = self.buildEqConstr(
                ego_x0, ego_up, dynamic_objects, obj_modes
            )
            runtime = time.time()-start_time   
            # print(f"build equality constraint takes {runtime}s")
            start_time = time.time()
            (
                GUxE,
                GUuE,
                LUE,
                GUxO,
                GUuO,
                LUO,
                GC,
                LC,
                constr_coll,
                GCS,
                LCS,
                static_constr_coll,
                GL,
                LL,
                constr_lane,
            ) = self.buildIneqConstr(
                ego_x0, ego_xp, obj_xp, ego_extent, dynamic_objects, static_objects, dyn_homotopy,static_homotopy, obj_modes, lane_info=lane_info
            )
            runtime = time.time()-start_time   
            # print(f"build inequality constraint takes {runtime}s")

            start_time = time.time()
            R, fu, Q_obj, Qr_ego,Qfr_ego, fx = self.buildCost(
                self.u0, ego_xp, dynamic_objects, obj_modes, xref,  self.joint_weights
            )
            runtime = time.time()-start_time   
            # print(f"build cost takes {runtime}s")
            
        # obj_x0 = np.stack([obj.x0 for obj in objects.values()], 0)
        
            if self.qp_prob is None:

                self.build_qp_solver_cvx(ego_x0[:1],ego_up[:1],ego_xp[:1],dynamic_objects,obj_xp,lane_info,dyn_homotopy[:1],GCS,LCS,R,fu,Q_obj, Qr_ego[:1],Qfr_ego[:1],fx[:1])

            Gu_ego = np.concatenate([GUxE,GUuE],-1)
            Gu_obj = np.concatenate([GUxO,GUuO],-1)
            
            GC,LC,GCS,LCS,GL,LL,Ge_ego,h_ego,Ge_obj,h_obj,Gu_ego,LUE,Gu_obj,LUO,Qr_ego,Qfr_ego,Q_obj,R,fx,fu,ego_xp,obj_xp,obj_up = TensorUtils.to_numpy((GC,LC,GCS,LCS,GL,LL,Ge_ego,h_ego,Ge_obj,h_obj,Gu_ego,LUE,Gu_obj,LUO,Qr_ego,Qfr_ego,Q_obj,R,fx,fu,ego_xp,obj_xp,obj_up))
            
            return GC,LC,GCS,LCS,GL,LL,Ge_ego,h_ego,Ge_obj,h_obj,Gu_ego,LUE,Gu_obj,LUO,Qr_ego,Qfr_ego,Q_obj,R,fx,fu,ego_xp,obj_xp,obj_up

        
    def generate_opt_matrices(
        self,
        ego_xp,
        ego_up,
        obj_xp,
        obj_up,
        Ge_ego,
        h_ego,
        Ge_obj,
        h_obj,
        GUxE,
        GUuE,
        LUE,
        GUxO,
        GUuO,
        LUO,
        GC,
        LC,
        constr_coll,
        GL,
        LL,
        constr_lane,
        R,
        fu,
        Q,
        fx,
        **kwargs
    ):
        totaldim = self.countx+self.countu
        start_time = time.time()
        bs = constr_coll.shape[0]
        constr_coll = constr_coll.reshape(bs,-1)
        # idx = (constr_coll<5).any(0)
        # GC = GC[:,idx]
        # LC = LC[:,idx]
        # constr_coll = constr_coll[:,idx]
        if GL is not None:
            G_safety = jnp.concatenate((GC,GL),1)
            L_safety = jnp.concatenate((LC,LL),1)
        else:
            G_safety = GC
            L_safety = LC
        
        
        # constr_coll = jnp.concatenate((constr_coll,constr_lane),-1)
        Ge,he,G,h,nslack = combine_matrices(
                                        totaldim,
                                        Ge_ego,
                                        h_ego,
                                        Ge_obj,
                                        h_obj,
                                        GUxE,
                                        GUuE,
                                        LUE,
                                        GUxO,
                                        GUuO,
                                        LUO,
                                        G_safety,
                                        L_safety,
                                        )
        
        x0 = np.concatenate((
            ego_xp.reshape(bs,-1),
            obj_xp.reshape(1,-1).repeat(bs,0),
            ego_up.reshape(bs,-1),
            obj_up.reshape(1,-1).repeat(bs,0),
            np.zeros([bs,nslack])
        ),-1)
        # G,LB,UB,Q,R,fx,fu = np.array(G),np.array(LB),np.array(UB),np.array(Q),np.array(R),np.array(fx),np.array(fu)
        P = [block_diag(Q[i],R,np.zeros([nslack,nslack])) for i in range(bs)]
        if GL is not None:
            fs = np.concatenate([np.ones(constr_coll.shape[1])*self.Mcoll,np.ones(constr_lane.shape[1])*self.Mlane])
        else:
            fs = np.ones(constr_coll.shape[1])*self.Mcoll
        f = np.concatenate([fx,fu[None,:].repeat(bs,0),fs[None,:].repeat(bs,0)],-1)

        


        # debug
        # obj_x0 = kwargs.get("obj_x0",None)
        # ego_x0 = kwargs.get("ego_x0",None)
        # Na = obj_x0.shape[0]
        # i=0
        # prob = osqp.OSQP()
        # prob.setup(P[i], f[i], csc_matrix(G[i]), LB[i], UB[i], alpha=1.0,verbose=False,max_iter=20000)
        # res = prob.solve()
        # xdim = 4
        # udim = 2
        # T = self.horizon
        # ego_x = x0[i,:xdim*T].reshape(T,xdim)
        # obj_x = x0[i,xdim*T:self.countx].reshape(-1,T,xdim)
        # ego_u = x0[i,self.countx:self.countx+udim*T].reshape(T,udim)
        # # for i,homotree in enumerate(list(homotopy_dict.keys())):
        # #     prob = osqp.OSQP()
        # #     prob.setup(P, f[i], csc_matrix(G[i]), LB[i], UB[i], alpha=1.0,verbose=False)
        # #     homotree.opt_prob = prob
        # dyn = UnicycleJax(0.1)
        # xx = np.array(dyn.forward_dynamics(ego_x[0],ego_u[1:],0.1)[0])
        # ll=(Ge_obj@x0[i,:1980,None]).squeeze(-1)-h_obj
        # Gu = np.concatenate([GUxE,GUuE],-1)
        # ll1=(Gu[i]@x0[i,:1980,None]).squeeze(-1)-LUE[i]
        
        # obj_x0 = kwargs.get("obj_x0",None)
        # ego_x0 = kwargs.get("ego_x0",None)
        # Na = obj_x0.shape[0]
        # ego_xp = dyn.forward_dynamics(ego_x0,ego_up*0,0.1)[0]
        # obj_xp = dyn.forward_dynamics(obj_x0,obj_up*0,0.1)[0]
        
        # x1=np.concatenate((ego_xp[0].flatten(),obj_xp.flatten(),ego_up[0].flatten()*0,obj_up.flatten()*0,np.ones(nslack)*1000),-1)
        # ll = (G[i]@x1[:,None]).squeeze(-1)-LB[i]
        # ll1 = UB[i]-(G[i]@x1[:,None]).squeeze(-1)
        # Gu_obj = np.concatenate([GUxO,GUuO],-1)
        runtime = time.time()-start_time   
        # print(f"batch run time is {runtime}s")
        # for i in range(5):
        #     prob = osqp.OSQP()
        #     prob.setup(P, f[i], csc_matrix(G[i]), LB[i], UB[i], alpha=1.0,verbose=False,max_iter=2000)
        #     prob.warm_start(x=x0[i])
        #     res = prob.solve()
        #     print(i)
        return P,f,Ge,he,G,h,nslack,x0



    def buildEqConstr(self, ego_x0, ego_up, objects,obj_modes):
        # Build matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: Gx*x+Gu*u = L
        # key = jax.random.PRNGKey(0)
        # ego_x0 = ego_x0+jax.random.normal(key)*0.1

        T = self.horizon
        
        ego_xp, Gx_ego,Gu_ego,L_ego = buildEgoEq(T,ego_x0,ego_up)
        bs = ego_x0.shape[0]
        
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        
        Gx_ego = jnp.concatenate((Gx_ego,jnp.zeros([bs,Gx_ego.shape[1],self.countx-xdim*T])),-1)
        Gu_ego = jnp.concatenate((Gu_ego,jnp.zeros([bs,Gu_ego.shape[1],self.countu-udim*T])),-1)

        # Sanity check
        # ll=(Gx_ego@ego_xp.reshape(bs,-1,1)+Gu_ego@ego_up.reshape(bs,-1,1)-L_ego[...,None]).squeeze(-1)

        # batch linearize dynamics for objects of the same type
        # TODO: prepare for varying xdim and udim
        
        Gx_obj = jnp.eye(self.countx - xdim * T)[None,:].repeat(bs,0)
        Gu_obj = jnp.zeros((bs,self.countx - xdim * T, self.countu - udim * T))
        L_obj = jnp.zeros((bs,self.countx - xdim * T))
        obj_xp = jnp.zeros([bs,len(objects), T, xdim])
        track_id_list = list(objects.keys())
        obj_up = jnp.zeros([bs,len(objects),T,udim])
        all_modes = jnp.unique(obj_modes)
        for nt, dyn in dynamic.items():
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:
                indices = jnp.array([track_id_list.index(id) for id in track_ids])
                x0 = np.stack([obj.x0 for _, obj in objects.items() if obj.type == nt], 0)[None,:].repeat(bs,0)
                up = np.stack([obj.ui for _, obj in objects.items() if obj.type == nt], 1)
                A1_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * (T - 1), self.ndx[track_id])
                        for track_id in track_ids
                    ]
                )
                A2_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * T, self.ndx[track_id] - xdim)
                        for track_id in track_ids
                    ]
                )

                B1_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * T, self.ndx[track_id])
                        for track_id in track_ids
                    ]
                )
                B2_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndu[track_id] - udim * T, self.ndu[track_id])
                        for track_id in track_ids
                    ]
                )

                L1_idx = jnp.hstack(
                    [
                        jnp.arange(
                            self.ndx[track_id] - xdim * T,
                            self.ndx[track_id] - xdim * (T - 1),
                        )
                        for track_id in track_ids
                    ]
                )


                obj_xp,obj_up,Gx_obj,Gu_obj,L_obj = buildObjEqBatch(nt,x0[0],up,indices,A1_idx,A2_idx,B1_idx,B2_idx,L1_idx,obj_xp,obj_up,Gx_obj,Gu_obj,L_obj)
           
        # sanity check
        # xx = np.concatenate([obj.xp.flatten() for _, obj in objects.items()], 0)
        # uu = np.concatenate([obj.up.flatten() for _, obj in objects.items()], 0)

        # ll = (Gx_obj @ xx[..., None] + Gu_obj @ uu[..., None]).squeeze(
        #     -1
        # ) - L_obj 
        # xu = jnp.concatenate([ego_xp[0].flatten(),jnp.zeros(256),ego_up[0].flatten(),jnp.zeros(128)])
        G_ego = jnp.concatenate([Gx_ego, Gu_ego], -1)
        Gx_obj = jnp.concatenate([jnp.zeros([bs,Gx_obj.shape[1],xdim*T]),Gx_obj],-1)
        Gu_obj = jnp.concatenate([jnp.zeros([bs,Gu_obj.shape[1],udim*T]),Gu_obj],-1)
        G_obj = jnp.concatenate([Gx_obj,Gu_obj],-1)
        # G_ego = [csc_matrix(m) for m in G_ego]
        return G_ego, L_ego, ego_xp, G_obj, L_obj, obj_xp,obj_up
    
    def buildEqConstr_stage(self, ego_x0, ego_up, objects,obj_modes):
        # Build matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
        # The equality constraint is: Gx*x+Gu*u = L
        # key = jax.random.PRNGKey(0)
        # ego_x0 = ego_x0+jax.random.normal(key)*0.1

        T = self.horizon
        
        ego_xp, A_e, B_e, C_e = LinearizationFunc["VEHICLE"](ego_x0, ego_up)
        bs = ego_x0.shape[0]
        
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim



        # batch linearize dynamics for objects of the same type
        # TODO: prepare for varying xdim and udim
        
        A_o = jnp.zeros([bs,T,len(objects),xdim,xdim])
        B_o = jnp.zeros([bs,T,len(objects),xdim,udim])
        C_o = jnp.zeros([bs,T,len(objects),xdim])
        obj_xp = jnp.zeros([bs,len(objects), T, xdim])
        track_id_list = list(objects.keys())
        obj_up = jnp.zeros([bs,len(objects),T,udim])
        for nt, dyn in dynamic.items():
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:
                indices = jnp.array([track_id_list.index(id) for id in track_ids])
                x0 = np.stack([obj.x0 for _, obj in objects.items() if obj.type == nt], 0)[None,:].repeat(bs,0)
                up = np.stack([obj.ui for _, obj in objects.items() if obj.type == nt], 1)
                obj_xp_nt, A_o_nt, B_o_nt, C_o_nt = LinearizationFunc[nt+"_batch"](x0, up)
                obj_xp = obj_xp.at[:,indices].set(obj_xp_nt)
                A_o = A_o.at[:,:,indices].set(A_o_nt.transpose(0,2,1,3,4))
                B_o = B_o.at[:,:,indices].set(B_o_nt.transpose(0,2,1,3,4))
                C_o = C_o.at[:,:,indices].set(C_o_nt.transpose(0,2,1,3))
        
        A = jnp.concatenate([A_e[:,:,None],A_o],2)
        B = jnp.concatenate([B_e[:,:,None],B_o],2)
        C = jnp.concatenate([C_e[:,:,None],C_o],2)

        return ego_xp,obj_xp,A,B,C


    def buildIneqConstr(
        self, ego_x0, ego_xp, dyn_obj_xp, ego_extent, dynamic_objects, static_objects, dyn_homotopy,static_homotopy, obj_modes, lane_info=None
    ):
        
        bs, T = ego_xp.shape[:2]
        # be careful of the state order change
        ego_lw = ego_extent[:2]
        dyn_track_id_list = list(dynamic_objects.keys())
        static_track_id_list = list(static_objects.keys())
        
        static_obj_xp = jnp.stack([obj.xp[obj_modes] for k, obj in static_objects.items()],1)
        static_obj_lw = jnp.stack([obj.extent[:2] for obj in static_objects.values()], 0)


        dyn_obj_lw = jnp.stack([obj.extent[:2] for obj in dynamic_objects.values()], 0)
        GC = list()
        LC = list()
        static_GC = list()
        static_LC = list()
        static_constr_coll = list()
        constr_coll = list()
        GUxO = list()
        GUuO = list()
        LUO = list()

        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        bs, T = ego_xp.shape[:2]
        
        GUxE,GUuE,LUE,GL,LL,constr_lane = buildEgoIneq(self.countx,self.countu,ego_x0,ego_xp,ego_lw,lane_info,self.cfg.ignore_heading_grad)
        # setup collision avoidance constraint and input constraint for objects
        for nt in collfun:
            track_ids = [
                track_id for track_id, obj in dynamic_objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:
                indices = jnp.array([dyn_track_id_list.index(id) for id in track_ids])
                dyn_obj_xp_nt, dyn_obj_lw_nt, dyn_homotopy_nt = (
                    dyn_obj_xp.take(indices, 1),
                    dyn_obj_lw.take(indices, 0),
                    dyn_homotopy.take(indices, 1),
                )

                
                dyn_obj_x0 = jnp.stack(
                    [obj.x0 for _, obj in dynamic_objects.items() if obj.type == nt], 0
                )
                Ox_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndx[track_id], self.ndx[track_id] + xdim * T)
                        for track_id in track_ids
                    ],
                    0,
                )
                Ou_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndu[track_id], self.ndu[track_id] + udim * T)
                        for track_id in track_ids
                    ],
                    0,
                )
                constr_coll_nt,GUx_nt,GUu_nt,LUO_nt,GC_nt,LC_nt = buildObjIneq(self.horizon,self.countx,self.countu,nt,dyn_obj_x0,dyn_obj_xp_nt, dyn_obj_lw_nt, dyn_homotopy_nt, ego_xp,ego_lw,Ox_indices,Ou_indices,self.cfg.ignore_heading_grad)
                
                
                
                GC.append(GC_nt)
                LC.append(LC_nt)
                GUxO.append(GUx_nt)
                GUuO.append(GUu_nt)
                LUO.append(LUO_nt)
                constr_coll.append(constr_coll_nt)
            
            # add constraints for static objects
            s_track_ids = [
                track_id for track_id, obj in static_objects.items() if obj.type == nt
            ]
            if len(s_track_ids)>0:
                
                indices = jnp.array([static_track_id_list.index(id) for id in s_track_ids])
                s_obj_xp, s_obj_lw, s_homotopy_nt = (
                    static_obj_xp.take(indices, 1),
                    static_obj_lw.take(indices, 0),
                    static_homotopy.take(indices, 1),
                )
                scc_nt,sGc_nt, sLc_nt = buildstaticEgoIneq(nt,s_obj_xp,s_obj_lw,ego_xp,ego_lw,s_homotopy_nt,self.cfg.ignore_heading_grad)

                
                static_GC.append(sGc_nt)
                static_LC.append(sLc_nt)
                static_constr_coll.append(scc_nt)
        GC = jnp.concatenate(GC, 1) if len(GC) > 0 else jnp.zeros([bs,0,self.countx])
        # GC = jnp.concatenate([GC,jnp.zeros([bs,GC.shape[1],self.countu])],-1)
        LC = jnp.concatenate(LC, 1) if len(LC) > 0 else jnp.zeros([bs,0])
        GCS = jnp.concatenate(static_GC, 1) if len(static_GC) > 0 else jnp.zeros([bs,0,self.countx])
        LCS = jnp.concatenate(static_LC, 1) if len(static_LC) > 0 else jnp.zeros([bs,0])
        GUxO = jnp.concatenate(GUxO, 1) if len(GUxO) > 0 else jnp.zeros([bs,0,self.countx])
        GUuO = jnp.concatenate(GUuO, 1) if len(GUuO) > 0 else jnp.zeros([bs,0,self.countu])
        LUO = jnp.concatenate(LUO, 1) if len(LUO) > 0 else jnp.zeros([bs,0])
        constr_coll = jnp.concatenate(constr_coll, 1) if len(constr_coll) > 0 else None
        static_constr_coll = jnp.concatenate(static_constr_coll, 1) if len(static_constr_coll) > 0 else None
        
        # sanity check
        # ego_xl = jnp.concatenate([ego_x0[:,None],ego_xp[:,:-1]],1)
        # lb,ub = self.ego_dyn.ubound(np.array(ego_xl))
        # LUE[0]-(GUxE[0,:,:T*xdim]@ego_xp[0].flatten()[:,None]).squeeze(-1)-(GUuE[0,:,:T*udim]@ub[0].flatten()[...,None]).squeeze(-1)

        # xx = jnp.concatenate([ego_xp[0].flatten(),dyn_obj_xp[0].flatten()],-1)
        # y1 = LC[0]-(GC[0]@xx[:,None]).squeeze(-1)
        # y2 = LCS[0]-(GCS[0]@ego_xp[0].flatten()[:,None]).squeeze(-1)
        return (
            GUxE,
            GUuE,
            LUE,
            GUxO,
            GUuO,
            LUO,
            GC,
            LC,
            constr_coll,
            GCS,
            LCS,
            static_constr_coll,
            GL,
            LL,
            constr_lane,
        )

    def buildIneqConstr_stage(
        self, ego_x0, ego_xp, dyn_obj_xp, ego_extent, dynamic_objects, static_objects, dyn_homotopy,static_homotopy, obj_modes, lane_info=None, sparsity=False,
    ):
        
        bs, T = ego_xp.shape[:2]
        xdim,udim=4,2
        # be careful of the state order change
        ego_lw = ego_extent[:2]
        dyn_track_id_list = list(dynamic_objects.keys())
        static_track_id_list = list(static_objects.keys())
        
        if self.num_static_mode==1:
            static_obj_xp = jnp.stack([obj.xp[obj_modes] for k, obj in static_objects.items()],1) if len(static_objects)>0 else jnp.zeros([0,T,xdim])
        else:
            static_obj_xp = jnp.stack([obj.xp[:self.num_static_mode] for k, obj in static_objects.items()],1) if len(static_objects)>0 else jnp.zeros([0,T,xdim])
            static_obj_xp = static_obj_xp[None].repeat(bs,0)
        static_obj_lw = jnp.stack([obj.extent[:2] for obj in static_objects.values()], 0) if len(static_objects)>0 else jnp.zeros([0,2])


        dyn_obj_lw = jnp.stack([obj.extent[:2] for obj in dynamic_objects.values()], 0) if len(dynamic_objects)>0 else jnp.zeros([0,2])
        GC = list()
        LC = list()
        static_GC = list()
        static_LC = list()
        static_constr_coll = list()
        constr_coll = list()
        GUxO = list()
        GUuO = list()
        LUO = list()
        Na = len(dynamic_objects)

        
        GUxE,GUuE,LUE,GL,LL,constr_lane = buildEgoIneq_stage(ego_x0,ego_xp,ego_lw,lane_info,self.cfg.ignore_heading_grad,sparsity)
        # setup collision avoidance constraint and input constraint for objects
        for nt in collfun:
            track_ids = [
                track_id for track_id, obj in dynamic_objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:

                indices = jnp.array([dyn_track_id_list.index(id) for id in track_ids])
                dyn_obj_xp_nt, dyn_obj_lw_nt, dyn_homotopy_nt = (
                    dyn_obj_xp.take(indices, 1),
                    dyn_obj_lw.take(indices, 0),
                    dyn_homotopy.take(indices, 1),
                )

                
                dyn_obj_x0 = jnp.stack(
                    [obj.x0 for _, obj in dynamic_objects.items() if obj.type == nt], 0
                )


                constr_coll_nt,GC_nt,LC_nt,GUx_nt,GUu_nt,LUO_nt = buildObjIneq_stage(self.horizon,nt,Na,dyn_obj_x0,dyn_obj_xp_nt, dyn_obj_lw_nt, dyn_homotopy_nt, ego_xp,ego_lw,indices,self.cfg.ignore_heading_grad,sparsity)
                
                
                
                GC.append(GC_nt)
                LC.append(LC_nt)
                GUxO.append(GUx_nt)
                GUuO.append(GUu_nt)
                LUO.append(LUO_nt)
                constr_coll.append(constr_coll_nt)
            
            # add constraints for static objects
            s_track_ids = [
                track_id for track_id, obj in static_objects.items() if obj.type == nt
            ]
            if len(s_track_ids)>0:
                indices = jnp.array([static_track_id_list.index(id) for id in s_track_ids])
                if self.num_static_mode==1:
                    s_obj_xp, s_obj_lw, s_homotopy_nt = (
                        static_obj_xp.take(indices, 1),
                        static_obj_lw.take(indices, 0),
                        static_homotopy.take(indices, 1),
                    )
                else:
                    s_obj_xp = static_obj_xp.take(indices,2).reshape(bs,-1,T,xdim)
                    s_obj_lw = jnp.tile(static_obj_lw.take(indices, 0),[self.num_static_mode,1])
                    s_homotopy_nt = jnp.tile(static_homotopy.take(indices, 1),[1,self.num_static_mode,1])
                scc_nt,sGc_nt, sLc_nt = buildstaticEgoIneq_stage(nt,s_obj_xp,s_obj_lw,ego_xp,ego_lw,s_homotopy_nt,self.cfg.ignore_heading_grad,sparsity)

                
                static_GC.append(sGc_nt)
                static_LC.append(sLc_nt)
                static_constr_coll.append(scc_nt)

        GC = jnp.concatenate(GC, 2) if len(GC)>0 else jnp.zeros([bs,T,0,(Na+1)*xdim])
        # GC = jnp.concatenate([GC,jnp.zeros([bs,GC.shape[1],self.countu])],-1)
        LC = jnp.concatenate(LC, 2) if len(LC)>0 else jnp.zeros([bs,T,0])
        GCS = jnp.concatenate(static_GC, 2) if len(static_GC)>0 else jnp.zeros([bs,T,0,xdim])
        LCS = jnp.concatenate(static_LC, 2) if len(static_LC)>0 else jnp.zeros([bs,T,0])
        GUxO = jnp.concatenate(GUxO, 2) if len(GUxO)>0 else jnp.zeros([bs,T,0,Na*xdim])
        GUuO = jnp.concatenate(GUuO, 2) if len(GUuO)>0 else jnp.zeros([bs,T,0,Na*udim])
        LUO = jnp.concatenate(LUO, 2) if len(LUO)>0 else jnp.zeros([bs,T,0])
        constr_coll = jnp.concatenate(constr_coll, 2) if len(constr_coll)>0 else None
        static_constr_coll = jnp.concatenate(static_constr_coll, 2) if len(static_constr_coll)>0 else None
        GUxE = jnp.concatenate([GUxE,jnp.zeros([bs,T,GUxE.shape[2],Na*xdim])],-1)
        GUuE = jnp.concatenate([GUuE,jnp.zeros([bs,T,GUuE.shape[2],Na*udim])],-1)

        GUxO = jnp.concatenate([jnp.zeros([bs,T,GUxO.shape[2],xdim]),GUxO],-1)
        GUuO = jnp.concatenate([jnp.zeros([bs,T,GUuO.shape[2],udim]),GUuO],-1)
        GU = jnp.concatenate([jnp.concatenate([GUxE,GUuE],-1),jnp.concatenate([GUxO,GUuO],-1)],-2)
        LU = jnp.concatenate([LUE,LUO],-1)

        return (
            GU,
            LU,
            GC,
            LC,
            constr_coll,
            GCS,
            LCS,
            static_constr_coll,
            GL,
            LL,
            constr_lane,
        )



    def buildCost(self, ego_u0, ego_xp, objects, obj_modes, xref, weights):
        T = self.horizon
        bs = ego_xp.shape[0]


        # Q and R
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim

        # along batch dimension, only xf_ref is different, Q and R are the same
        R_ego = self.R_ego
        Qr_ego = jnp.zeros_like(self.Qr_ego)[None,:].repeat(bs,0)
        Qfr_ego = jnp.zeros_like(self.Qfr_ego)[None,:].repeat(bs,0)

        dR_ego = self.dR_ego
        S = self.S
        fx_ego = jnp.zeros([bs, xdim * T])
        fu_ego = jnp.zeros(udim * T)

        
        
        
        if xref is not None:

            Qr_ref,Qfr_ref,fref = buildXrefCost_Qr(T,bs,xref,self.Qr_ego,self.Qfr_ego)
            Qr_ego+= np.sqrt(weights["ego"])*Qr_ref
            Qfr_ego+= np.sqrt(weights["ego"])*Qfr_ref
            fx_ego+= weights["ego"]*fref
            
            
        # add the du penalty on the first u
        
        fu_ego = fu_ego.at[:udim].set(-self.drego @ ego_u0)
        R = jnp.zeros([self.countu, self.countu])

        fu = jnp.zeros(self.countu)
        fx = jnp.zeros([bs,self.countx])

        R = R.at[: udim * T, : udim * T].add((dR_ego + R_ego) * weights["ego"])
        fu = fu.at[: udim * T].set(fu_ego * weights["ego"])
        fx = fx.at[:,:xdim*T].add(fx_ego*weights["ego"])
        Q_obj = jnp.zeros([self.countx, self.countx]) 
        for nt in dynamic:
            xdim = dynamic[nt].xdim
            udim = dynamic[nt].udim
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:

                num_agent = len(track_ids)
                # u0_nt = jnp.stack([obj.u0 for _, obj in objects.items() if obj.type == nt])
                xp_nt = jnp.stack(
                    [obj.xp[obj_modes] for _, obj in objects.items() if obj.type == nt]
                ,1).reshape(bs,-1, xdim * T)
                u_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndu[track_id], self.ndu[track_id] + udim * T)
                        for track_id in track_ids
                    ]
                )
                # u0_indices = jnp.concatenate(
                #     [
                #         jnp.arange(self.ndu[track_id], self.ndu[track_id] + udim)
                #         for track_id in track_ids
                #     ]
                # )
                x_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndx[track_id], self.ndx[track_id] + xdim * T)
                        for track_id in track_ids
                    ]
                )

                Su = jnp.kron(S, jnp.eye(udim))
                dR_nt = Su.T @ np.kron(np.eye(T - 1), self.dR[nt]) @ Su

                # add the du penalty on the first u
                # dR_nt = dR_nt.at[:udim, :udim].add(self.dR[nt])
                dR_nt_mat = jnp.kron(jnp.eye(num_agent), dR_nt)
                R_nt = jnp.kron(jnp.eye(T), self.R[nt])
                R_nt_mat = jnp.kron(jnp.eye(num_agent), R_nt)
                Q_nt = jnp.kron(jnp.eye(T), self.Q[nt])
                Q_nt_mat = jnp.kron(jnp.eye(num_agent), Q_nt)
                Lx_nt = -(Q_nt[None] @ xp_nt[..., None]).reshape(bs,-1)

                R = R.at[np.ix_(u_indices, u_indices)].add((R_nt_mat + dR_nt_mat) * weights["obj"])
                # fu = fu.at[u0_indices].set((
                #     -self.dR[nt][None, :] @ u0_nt[..., None]
                # ).flatten() * weights["obj"])
                Q_obj = Q_obj.at[np.ix_(x_indices, x_indices)].set(Q_nt_mat * weights["obj"])
                fx = fx.at[:,x_indices].set(Lx_nt * weights["obj"])
        return R, fu, Q_obj, Qr_ego,Qfr_ego, fx
    
    def buildCost_stage(self, ego_u0, ego_xp, objects, obj_modes, xref, weights):
        T = self.horizon
        bs = ego_xp.shape[0]


        # Q and R
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        Na = len(objects)

        # along batch dimension, only xf_ref is different, Q and R are the same

        fu_ego = jnp.zeros([T,udim])
        fu_ego = fu_ego.at[0].set(-self.drego @ ego_u0)

        Q = jnp.zeros([bs,T,Na+1,xdim,xdim])
        R = jnp.zeros([T,Na+1,udim,udim])
        dR = jnp.zeros([T,Na+1,udim,udim])

        fu = jnp.zeros([T,Na+1,udim])
        fx = jnp.zeros([bs,T,Na+1,xdim])
        fu = fu.at[: 0].set(fu_ego * weights["ego"])
        
        
        if xref is not None:
            Q_ref,fref = buildXrefCost_stage(T,bs,xref,self.qego,self.Qf_ego,self.cfg.rot_Q)
            Q = Q.at[:,:,0].set(weights["ego"]*Q_ref)
            fx = fx.at[:,:,0].set(weights["ego"]*fref)

            
            
        # add the du penalty on the first u
        


        R = R.at[:,0].add(self.rego[None,:].repeat(T,0)* weights["ego"])
        dR = dR.at[:,0].add(self.drego[None].repeat(T,0)* weights["ego"])
        

        track_id_list = list(objects.keys())
        for nt in dynamic:
            xdim = dynamic[nt].xdim
            udim = dynamic[nt].udim
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:

                num_agent = len(track_ids)
                # u0_nt = jnp.stack([obj.u0 for _, obj in objects.items() if obj.type == nt])
                xp_nt = jnp.stack(
                    [obj.xp[obj_modes] for _, obj in objects.items() if obj.type == nt]
                ,2)

                indices = jnp.array([track_id_list.index(id) for id in track_ids])
                R = R.at[:,indices+1].add(self.R[nt][None,None].repeat(T,0).repeat(indices.shape[0],1)* weights["obj"])
                dR = dR.at[:,indices+1].add(self.dR[nt][None,None].repeat(T,0).repeat(indices.shape[0],1)* weights["obj"])
                Q = Q.at[:,:,indices+1].add(self.Q[nt][None,None,None].repeat(bs,0).repeat(T,1).repeat(indices.shape[0],2)* weights["obj"])
                fx = fx.at[:,:,indices+1].add(-(self.Q[nt][None,None,None]@xp_nt[...,None]).squeeze(-1)*weights["obj"])

        return R, fu, Q, fx, dR
    

    
    def get_xref_from_lane(self,lane,vdes,xyh=np.zeros(3)):
        delta_x,_,_ = GeoUtils.batch_proj(xyh,lane)
        try:
            idx = np.abs(delta_x).argmin()
            s = np.linalg.norm(lane[idx+1:,:2]-lane[idx:-1,:2],axis=-1,keepdims=True).cumsum()
            s = np.insert(s,0,0.0)-delta_x[idx]
            f = interp1d(s,lane[idx:],axis=0,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
        except:
            f = interp1d(-delta_x,lane,axis=0,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
        t = np.arange(1,self.horizon+1)*self.dt
        xref = f(vdes*t)
        xref = np.concatenate((xref[:,:2],vdes*np.ones([self.horizon,1]),xref[:,2:]),-1)
        # if xref[0,0]>10:
        #     print("error")
        if np.isnan(xref).any():
            breakpoint()
        return xref
    
    def build_qp_solver_forcespro(self,ego_xp,obj_xp,A,B,C,GU,LU,GC,LC,GCS,LCS,GL,LL,R,fu,Q,fx,dR):
        
        self.param_sparsity = dict()
        slack_strat = self.cfg.slack_strat
        T = self.horizon 
        Na = obj_xp.shape[1]
        xdim,udim=4,2
        NC = GC.shape[2]
        NCS = GCS.shape[2]
        NL = GL.shape[2]
        if slack_strat == "Linf":
            NS = 2
        elif slack_strat == "L1":
            NS = NC+NCS+NL

        stages = forcespro.MultistageProblem(T+1) # 0-indexed
        # set up dimensions
        vardim = (Na+1)*(xdim+2*udim)+NS
        for i in range(T+1):
            stages.dims[i]['n'] = (Na+1)*(xdim+2*udim)+NS # length of stage variable zi
            stages.dims[i]['r'] = (Na+1)*(xdim+udim) # number of equality constraints
            stages.dims[i]['l'] = NS # number of lower bounds
            stages.dims[i]['u'] = 0 # number of upper bounds
            stages.dims[i]['p'] = NC+NCS+NL+GU.shape[2] # number of polytopic constraints
            stages.dims[i]['q'] = 0 # number of quadratic constraints
        
        # declare parameters

        # eq.C
        # Aone = np.ones_like(A[0,0])
        # Bone = np.ones_like(B[0,0])
        # Aone = TensorUtils.block_diag_from_cat_jit(Aone)
        # Bone = TensorUtils.block_diag_from_cat_jit(Bone)
        # eyeu = np.eye((Na+1)*udim)

        # Gx = np.concatenate([np.concatenate([Aone,Bone,np.zeros_like(Bone)],-1),],-2)
        # Gu = np.concatenate([np.zeros([udim*(Na+1),xdim*(Na+1)]),eyeu,np.zeros_like(eyeu)],-1)
        # Cstruct = np.concatenate([Gx,Gu],-2)
        # # Cvar = np.concatenate([Gx,0*Gu],-2)
        # Cstruct = np.concatenate([Cstruct,np.zeros([Cstruct.shape[0],NS])],-1)
        
        
        
        # Cstruct = np.concatenate([np.ones])
        # self.param_sparsity["eq.C"] = (Cstruct!=0).astype(float)
        
        for i in range(1,T+1):
            # stages.newParam('EC_'+str(i),[i],'eq.C','sparse',Cstruct)
            stages.newParam('EC_'+str(i),[i],'eq.C')


        # eq.D
        D = np.concatenate([np.concatenate([-np.eye(xdim*(Na+1)),np.zeros([xdim*(Na+1),2*udim*(Na+1)+NS])],-1),
            np.concatenate([np.zeros([udim*(Na+1),xdim*(Na+1)]),-np.eye((Na+1)*udim),self.dt*np.eye((Na+1)*udim),np.zeros([udim*(Na+1),NS])],-1)],-2)
        for i in range(T+1):
            stages.eq[i]['D'] = D

        stages.newParam("xinit", [1], 'eq.c') # 1-indexed
        # eq.c
        for i in range(2,T+2):
            stages.newParam('Ec_'+str(i),[i],'eq.c')

        # ineq.p.A
        if slack_strat == "L1":
            gc = np.concatenate([GC[0,0],np.zeros([NC,(Na+1)*2*udim]),-np.eye(NC),np.zeros([NC,NCS+NL])],-1)
            gcs = np.concatenate([GCS[0,0],np.zeros([NCS,Na*xdim+(Na+1)*2*udim]),np.zeros([NCS,NC]),-np.eye(NCS),np.zeros([NCS,NL])],-1)
            gl = np.concatenate([GL[0,0],np.zeros([NL,vardim-xdim-NL]),-np.eye(NL)],-1)
        elif slack_strat == "Linf":
            gc = np.concatenate([GC[0,0],np.zeros([NC,(Na+1)*2*udim]),-np.ones([NC,1]),np.zeros([NC,1])],-1)
            gcs = np.concatenate([GCS[0,0],np.zeros([NCS,Na*xdim+(Na+1)*2*udim]),-np.ones([NCS,1]),np.zeros([NCS,1])],-1)
            gl = np.concatenate([GL[0,0],np.zeros([NL,vardim-xdim-1]),-np.ones([NL,1])],-1)
        gu = np.concatenate([GU[0,0],np.zeros([GU.shape[2],NS+(Na+1)*udim])],-1)
        Amat = np.concatenate([gc,gcs,gl,gu],-2)
        Amat=Amat.flatten().reshape(*Amat.shape,order="F")
        Avar = (Amat!=0).astype(float)
        # Avar[:,-NS:] = 0
        self.param_sparsity["ineq.p.A"] = Avar
        for i in range(1,T+2):
            # stages.newParam('pA_'+str(i),[i],'ineq.p.A','sparse',Avar,Avar)
            stages.newParam('pA_'+str(i),[i],'ineq.p.A')

        # ineq.p.b
        for i in range(1,T+2):
            stages.newParam('pb_'+str(i),[i],'ineq.p.b')

        # cost.H
        q = TensorUtils.block_diag_from_cat_jit(Q[0,0])
        r = TensorUtils.block_diag_from_cat_jit(R[0])
        dr = TensorUtils.block_diag_from_cat_jit(dR[0])
        
        # Hstruct = block_diag(q,r,dr,np.zeros([NS,NS]))[None].repeat(T+1,0)
        # self.param_sparsity["cost.H"] = (Hstruct!=0)
        H = np.zeros([vardim,vardim])
        H[:xdim*(Na+1),:xdim*(Na+1)] = q
        H[xdim*(Na+1):xdim*(Na+1)+udim*(Na+1),xdim*(Na+1):xdim*(Na+1)+udim*(Na+1)] = r
        H[xdim*(Na+1)+udim*(Na+1):xdim*(Na+1)+udim*(Na+1)+udim*(Na+1),xdim*(Na+1)+udim*(Na+1):xdim*(Na+1)+udim*(Na+1)+udim*(Na+1)] = dr
        for i in range(0,T+1):
            stages.cost[i]['H'] = H

                    

        # cost.f
        # fstruct = np.concatenate([np.ones([T,xdim]),np.ones([T,udim]),np.ones([T,udim]),np.ones([T,NC+NCS])*self.Mcoll,np.ones([T,NL])*self.Mlane],-1)
        # fvar = np.concatenate([np.ones([T,xdim]),np.ones([T,udim]),np.ones([T,udim]),np.zeros([T,NS])],-1)
        # self.param_sparsity["cost.f"] = fvar
        for i in range(1,T+2):
            stages.newParam('f_'+str(i),[i],'cost.f')

        for i in range(T+1):
            stages.ineq[i]['b']['lbidx'] = np.arange(vardim-NS+1,vardim+1) # index vector for lower bounds, 1-indexed
            stages.ineq[i]['b']['lb'] = np.zeros(NS)    # lower bounds

        # declare output
        stages.newOutput("ego_x", range(2,T+2), range(1,xdim+1))
        stages.newOutput("ego_u", range(1,T+1), range((Na+1)*xdim+1,(Na+1)*xdim+udim+1))
        stages.newOutput("ego_du", range(1,T+1), range((Na+1)*xdim+(Na+1)*udim+1,(Na+1)*xdim+(Na+1)*udim+udim+1))
        if Na>0:
            stages.newOutput("obj_x", range(2,T+2), range(xdim+1,(Na+1)*xdim+1))
            stages.newOutput("obj_u", range(1,T+1), range((Na+1)*xdim+udim+1,(Na+1)*xdim+(Na+1)*udim+1))
            stages.newOutput("obj_du", range(1,T+1), range((Na+1)*xdim+(Na+1)*udim+udim+1,(Na+1)*(xdim+2*udim)+1))
        if slack_strat=="L1":
            if Na>0:
                stages.newOutput("Cslack", range(2,T+2), range((Na+1)*(xdim+2*udim)+1,(Na+1)*(xdim+2*udim)+NC+1))
            stages.newOutput("CSslack", range(2,T+2), range((Na+1)*(xdim+2*udim)+NC+1,(Na+1)*(xdim+2*udim)+NC+NCS+1))
            stages.newOutput("Lslack", range(2,T+2), range((Na+1)*(xdim+2*udim)+NC+NCS+1,vardim+1))
        elif slack_strat=="Linf":
            stages.newOutput("slack", range(2,T+2), range((Na+1)*(xdim+2*udim)+1,vardim+1))
        options = forcespro.CodeOptions(self.solvername)
        stages.codeoptions = options
        stages.codeoptions.server = 'https://forces-beta-cmplr.embotech.com'
        stages.codeoptions.optlevel=2
        stages.codeoptions.printlevel=0
        stages.codeoptions.threadSafeStorage = True
        stages.codeoptions.overwrite = 1
        stages.codeoptions.BuildSimulinkBlock = 0
        stages.codeoptions.cleanup = 0
        stages.codeoptions.parallel = 1
        # stages.codeoptions.sse = 1
        # stages.codeoptions.avx = 1
        
        stages.generateCode(get_userid.userid)
        
        print("Forces Pro code generated!")
        with open(self.solvername+"/sparsity.pkl","wb") as f:
            pickle.dump(self.param_sparsity,f)
        self.qp_prob = __import__(self.solvername+"_py")
        


    
    def build_qp_solver_cvx(self,ego_x0,ego_up,ego_xp,objects,obj_xp,lane_info,homotopy,GCS,LCS,R,fu,Q_obj, Qr_ego,Qfr_ego,fx,encode_sparsity = True):
        
        
        xdim = 4
        udim = 2
        T = self.horizon
        Nxe = xdim*T
        Nue = udim*T
        Ge_ego, Ge_obj,GUxE,GUuE,GUxO,GUuO,GC,GL = self.get_constr_sparsity_pattern(ego_x0,ego_up,ego_xp,objects,lane_info,homotopy)
        Ge_ego, Ge_obj,GUxE,GUuE,GUxO,GUuO,GC,GL,R,fu,Q_obj, Qr_ego,Qfr_ego,fx = TensorUtils.to_numpy((Ge_ego, Ge_obj,GUxE,GUuE,GUxO,GUuO,GC,GL,R,fu,Q_obj, Qr_ego,Qfr_ego,fx))

        xe = cp.Variable(Nxe, name='xe')
        ue = cp.Variable(Nue, name='ue')
        
        
        
        Na = obj_xp.shape[1]
        Nxo = Na*T*xdim
        Nuo = Na*T*udim
        if Na>0:
            xo = cp.Variable(Nxo, name='xo')
            uo = cp.Variable(Nuo, name='uo')

            xx = cp.hstack([xe,xo])
            uu = cp.hstack([ue,uo])
            xu = cp.hstack([xx,uu])
        else:
            xu = cp.hstack([xe,ue])
        
        
        Mc = GC[0].shape[0]
        Mcs = GCS[0].shape[0]
        sparsity = np.nonzero(GC[0])
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Gc = cp.Parameter(GC[0].shape,name="Gc",sparsity=sparsity)
        else:
            Gc = cp.Parameter(GC[0].shape,name="Gc")
        
        Lc = cp.Parameter(Mc,name="Lc")
        
        Gcs = cp.Parameter(GCS[0].shape,name="Gcs")
        Lcs = cp.Parameter(Mcs,name="Lcs")
        
        if self.cfg.slack_strat=="L1":
            Cslack = cp.Variable(Mc+Mcs,name="Cslack")
            Lslack = cp.Variable(GL[0].shape[0],name="Lslack")
        else:
            Cslack = cp.Variable(1,name="Cslack")
            Lslack = cp.Variable(1,name="Lslack")
        
        sparsity = np.nonzero(GL[0])
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Gl = cp.Parameter(GL[0].shape,name="Gl",sparsity=sparsity)
        else:
            Gl = cp.Parameter(GL[0].shape,name="Gl")
        
        Ll = cp.Parameter(GL[0].shape[0],name="Ll")
        
        
        sparsity = np.nonzero(Ge_ego[0])
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Gee = cp.Parameter(Ge_ego[0].shape, name='Gee', sparsity=sparsity)
        else:
            Gee = cp.Parameter(Ge_ego[0].shape, name='Gee')
        he = cp.Parameter(Ge_ego[0].shape[0], name='he')
        
        sparsity = np.nonzero(Ge_obj)
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Geo = cp.Parameter(Ge_obj.shape, name='Geo', sparsity=sparsity)
        else:
            Geo = cp.Parameter(Ge_obj.shape, name='Geo')
        ho = cp.Parameter(Ge_obj.shape[0], name='ho')
        
        
        Gu_ego = np.concatenate([GUxE[0],GUuE[0]],-1)
        Gu_obj = np.concatenate([GUxO,GUuO],-1)
        

        sparsity = np.nonzero(Gu_ego)
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Gue = cp.Parameter(Gu_ego.shape, name='Gue',sparsity = sparsity)
        else:
            Gue = cp.Parameter(Gu_ego.shape, name='Gue')
        Lue = cp.Parameter(Gu_ego.shape[0], name='Lue')
        
        sparsity = np.nonzero(Gu_obj[0])
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Guo = cp.Parameter(Gu_obj[0].shape, name='Guo',sparsity = sparsity)
        else:
            Guo = cp.Parameter(Gu_obj[0].shape, name='Guo')
        Luo = cp.Parameter(Gu_obj[0].shape[0], name='Luo')
        
        sparsity = np.nonzero(np.array(Qr_ego[0]))
        sparsity = [(a,b) for a,b in zip(sparsity[0],sparsity[1])]
        if encode_sparsity:
            Qr_ego = cp.Parameter(Qr_ego.shape[1:], name='Qr_ego',sparsity=sparsity)
        else:
            Qr_ego = cp.Parameter(Qr_ego.shape[1:], name='Qr_ego')

        
        Qfr_ego = cp.Parameter(Qfr_ego.shape[1:],name="Qfr_ego")
        
        
        fx = cp.Parameter(Nxe+Nxo, name='fx')
        fu = cp.Parameter(Nue+Nuo, name='fu')
        
        
        objective = (1/2)*(cp.quad_form(xx, Q_obj)+cp.sum_squares(Qr_ego@xe)+cp.sum_squares(Qfr_ego@xe[-xdim:])+cp.quad_form(uu,R)) + fx.T@xx + fu.T@uu + self.Mcoll*cp.sum(Cslack)+ self.Mlane*cp.sum(Lslack)
        if self.cfg.slack_strat=="L1":
            self.qp_prob = cp.Problem(cp.Minimize(objective),
                    [Gee @ xu == he,
                    Geo @ xu == ho,
                    Gc @ xx - Cslack[:Mc] <= Lc,
                    Gcs @ xe - Cslack[Mc:] <= Lcs,
                    Gl @ xe - Lslack <= Ll,
                    Gue @ xu <= Lue,
                    Guo @ xu <= Luo,
                    Cslack>= 0,
                    Lslack>= 0],
                    )
        elif self.cfg.slack_strat=="Linf":
            self.qp_prob = cp.Problem(cp.Minimize(objective),
                    [Gee @ xu == he,
                    Geo @ xu == ho,
                    Gc @ xx - Cslack <= Lc,
                    Gcs @ xe - Cslack <= Lcs,
                    Gl @ xe - Lslack <= Ll,
                    Gue @ xu <= Lue,
                    Guo @ xu <= Luo,
                    Cslack>= 0,
                    Lslack>= 0],
                    )

        if self.cfg.code_gen:
            cpg.generate_code(self.qp_prob, code_dir=str(self.qp_solver_dir), solver=self.cfg.qp_solver)
            cpg_solver = importlib.import_module(self.module_path+'.cpg_solver')
            self.qp_prob.register_solve('cpg', cpg_solver.cpg_solve)
        else:
            import pickle
            with open(str(self.qp_solver_dir / 'problem.pickle'), 'wb') as f:
                pickle.dump(self.qp_prob, f)

            

                
                                
        
        
        
    def get_constr_sparsity_pattern(self,ego_x0,ego_up,ego_xp,objects,lane_info,homotopy):
        # sparsity pattern for equality constraint
        T = self.horizon
        
        Gx_ego,Gu_ego = EgoEqSparsity(T,ego_x0,ego_up)
        bs = ego_x0.shape[0]
        
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        
        Gx_ego = jnp.concatenate((Gx_ego,jnp.zeros([bs,Gx_ego.shape[1],self.countx-xdim*T])),-1)
        Gu_ego = jnp.concatenate((Gu_ego,jnp.zeros([bs,Gu_ego.shape[1],self.countu-udim*T])),-1)

        
        Gx_obj = jnp.eye(self.countx - xdim * T)
        Gu_obj = jnp.zeros((self.countx - xdim * T, self.countu - udim * T))
        obj_xp = jnp.zeros([bs,len(objects), T, xdim])
        track_id_list = list(objects.keys())
        for nt, dyn in dynamic.items():
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:

                indices = jnp.array([track_id_list.index(id) for id in track_ids])
                x0 = np.stack([obj.x0 for _, obj in objects.items() if obj.type == nt], 0)
                up = np.stack([obj.up[0] for _, obj in objects.items() if obj.type == nt], 0)
                A1_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * (T - 1), self.ndx[track_id])
                        for track_id in track_ids
                    ]
                )
                A2_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * T, self.ndx[track_id] - xdim)
                        for track_id in track_ids
                    ]
                )

                B1_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndx[track_id] - xdim * T, self.ndx[track_id])
                        for track_id in track_ids
                    ]
                )
                B2_idx = jnp.hstack(
                    [
                        jnp.arange(self.ndu[track_id] - udim * T, self.ndu[track_id])
                        for track_id in track_ids
                    ]
                )

                
                Gx_obj,Gu_obj = ObjEqSparsity(nt,x0,up,A1_idx,A2_idx,B1_idx,B2_idx,Gx_obj,Gu_obj)
        
        G_ego = jnp.concatenate([Gx_ego, Gu_ego], -1)
        Gx_obj = jnp.concatenate([jnp.zeros([Gx_obj.shape[0],xdim*T]),Gx_obj],-1)
        Gu_obj = jnp.concatenate([jnp.zeros([Gu_obj.shape[0],udim*T]),Gu_obj],-1)
        G_obj = jnp.concatenate([Gx_obj,Gu_obj],-1)
        
        # sparsity pattern for inequality constraint
        ego_lw = jnp.ones(2)
        track_id_list = list(objects.keys())

        obj_lw = jnp.stack(
            [obj.extent[:2] for obj in objects.values() if obj.type == "VEHICLE"], 0
        )
        GC = list()


        GUxO = list()
        GUuO = list()
        xdim = self.ego_dyn.xdim
        udim = self.ego_dyn.udim
        bs, T = ego_xp.shape[:2]
        
        GUxE,GUuE,GL = EgoIneqSparsity(self.countx,self.countu,ego_x0,ego_xp,ego_lw,lane_info)
        # setup collision avoidance constraint and input constraint for objects
        for nt in collfun:
            track_ids = [
                track_id for track_id, obj in objects.items() if obj.type == nt
            ]
            if len(track_ids) > 0:

                indices = jnp.array([track_id_list.index(id) for id in track_ids])
                obj_xp_nt, obj_lw_nt, homotopy_nt = (
                    obj_xp.take(indices, 1),
                    obj_lw.take(indices, 0),
                    homotopy.take(indices, 1),
                )

                
                obj_x0_nt = jnp.stack(
                    [obj.x0 for _, obj in objects.items() if obj.type == nt], 0
                )
                Ox_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndx[track_id], self.ndx[track_id] + xdim * T)
                        for track_id in track_ids
                    ],
                    0,
                )
                Ou_indices = jnp.concatenate(
                    [
                        jnp.arange(self.ndu[track_id], self.ndu[track_id] + udim * T)
                        for track_id in track_ids
                    ],
                    0,
                )
                GUx_nt,GUu_nt,GC_nt = ObjIneqSparsity(self.horizon,self.countx,self.countu,nt,obj_x0_nt,obj_xp_nt,obj_lw_nt,homotopy_nt,ego_xp,ego_lw,Ox_indices,Ou_indices)

            

                GC.append(GC_nt)
                GUxO.append(GUx_nt)
                GUuO.append(GUu_nt)
        GC = jnp.concatenate(GC, 1)
        GUxO = jnp.concatenate(GUxO, 0)
        GUuO = jnp.concatenate(GUuO, 0)
        
        # get sparsity pattern for cost
        return G_ego, G_obj,GUxE,GUuE,GUxO,GUuO,GC,GL


    def solve_mpc(self, time_stamp, obs, pred_x, pred_u,lane_info = None, **kwargs):
        device = pred_x.device
        x0 = pred_x[..., 0, :]
        pred_x = pred_x[..., 1:, :]
        u0 = pred_u[..., 0, :]
        pred_u = pred_u[..., 1:, :]
        if self.dt!=self.pred_dt:
            simt = self.pred_dt*np.arange(1,pred_x.shape[-2]+1)
            mpct = self.dt*np.arange(1,self.horizon+1)
            f = interp1d(simt,TensorUtils.to_numpy(pred_x),axis=-2,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            pred_x = TensorUtils.to_torch(f(mpct),device=device)
            f = interp1d(simt,TensorUtils.to_numpy(pred_u),axis=-2,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            pred_u = TensorUtils.to_torch(f(mpct),device=device)
        vel = obs["curr_speed"][0]
        
        
        vdes = TensorUtils.to_numpy(torch.clip(vel+1,min=2.0,max=10.0))
        if "ego_xref" in kwargs:
            ego_xref = kwargs["ego_xref"]
        else:
            if lane_info is not None:
                center_lane = lane_info["center"]
                ego_xref = self.get_xref_from_lane(center_lane,vdes)
            else:
                ego_xref = None
        ego_goal = ego_xref[-1] if ego_xref is not None else None

        self.update_road_objects(
            time_stamp,
            x0[0],
            u0,
            pred_x,
            pred_u,
            obs["track_id"][1:],
            obs["agent_type"][1:],
            obs["extent"][1:],
        )
        vel = torch.clip(vel,self.ego_sampler.vbound[0]+1.0,self.ego_sampler.vbound[1]-1.0)
        traj0 = torch.tensor([[0.0, 0.0, vel, 0, 0.0, 0.0, 0.0]]).to(vel.device)

        def expand_func(x):
            return self.ego_sampler.gen_trajectory_batch(
                x,
                self.horizon_sec,
                lanes=None,
                N=self.horizon / self.stage + 1,
                max_children=5,
            )

        root = TrajTree(traj0, None, 0)
        root.grow_tree(expand_func, self.stage)
        nodes, _ = TrajTree.get_nodes_by_level(root, depth=self.stage)
        ego_trajs_full = torch.stack(
            [leaf.total_traj for leaf in nodes[self.stage]], 0
        ).float()[:, 1:, TRAJ_INDEX]
        Ne = ego_trajs_full.shape[0]
        # agent_pos = pred_x.unsqueeze(0).repeat_interleave(Ne, 0)[..., :2]
        # dis = torch.norm(ego_trajs_full[:, None, None, :, :2] - agent_pos, dim=-1)
        # dis = dis.min(-1)[0].min(1)[0].min(0)[0]
        dis = torch.norm(TensorUtils.to_torch(ego_xref[None,None,...,:2],device=device)-pred_x[...,:2],dim=-1).min(0)[0].min(-1)[0]
        dis[obs["agent_type"][1:]<=0]=np.inf
        if dis.shape[0]>=self.num_dynamic_object+self.num_static_object:
            idx = torch.argsort(dis)[: self.num_dynamic_object + self.num_static_object]

            # idx = torch.where(dis < self.cfg.distance_threshold)[0]
            pred_x = pred_x[:,idx]
            pred_u = pred_u[:,idx]

            obj_idx = obs["track_id"][idx+1]
            agent_extents = obs["extent"][obj_idx]
            agent_types = obs["agent_type"][obj_idx]
            
        else:
            idx = torch.argsort(dis)
            obj_idx = idx+1
            agent_extents = obs["extent"][obj_idx]
            agent_types = obs["agent_type"][obj_idx]
            
            obj_idx = torch.cat([obj_idx,-torch.arange(1,self.num_dynamic_object + self.num_static_object-dis.shape[0]+1,device=obj_idx.device)],0)
        dynamic_track_ids = obj_idx[:self.num_dynamic_object]
        static_track_ids = obj_idx[self.num_dynamic_object:self.num_dynamic_object+self.num_static_object]
        cost_weight = {
            "collision_weight": 0.0,
            "lane_weight": 1.0,
            "likelihood_weight": 0.20,
            "goal_reaching": 1.0
        }
        ego_extents = obs["extent"][0]
        ego_xyh = torch.cat(
            [Unicycle.state2pos(ego_trajs_full), Unicycle.state2yaw(ego_trajs_full)], -1
        )
        pred_xyh = torch.cat(
            [Unicycle.state2pos(pred_x), Unicycle.state2yaw(pred_x)], -1
        )
        sample_cost = self.sampling_cost_fun(
            cost_weight, ego_xyh, pred_xyh, ego_extents, agent_extents, agent_types, ego_goal, lane_info = lane_info
        )

        homotopy, ego_trajs, obj_modes, sample_cost = self.select_homotopy(ego_trajs_full, pred_x, sample_cost)
        # if pred_x.shape[1]==0:
        #     print("no agents")

        

        # From this point on, switch to JAX and Num
        vel = TensorUtils.to_numpy(vel)
        
        # for now, only check the top k homotopy candidates based on
        if sample_cost.shape[0]>=self.cfg.homo_candiate_num:

            _, idx_select = (-sample_cost).topk(self.cfg.homo_candiate_num)
            
            idx_select = TensorUtils.to_numpy(idx_select)
            homotopy = homotopy[idx_select]
            ego_trajs = ego_trajs[idx_select]
            obj_modes = obj_modes[idx_select]
        
        del sample_cost
        
        # homotopy_tree = grouping_homotopy_to_tree(homotopy, ego_trajs, None)
        # ego_xp_dict = OrderedDict()
        # ego_xp_dict,_ = depth_first_traverse(homotopy_tree,traverse_pick_first_ego_xp, {}, ego_xp_dict)
        # homotopy_dict = OrderedDict()
        # homotopy_dict,_ = depth_first_traverse(homotopy_tree,traverse_obtain_homotopy, {}, homotopy_dict)
        Ne = ego_trajs.shape[0]
        ego_x0 = Unicycle.combine_to_state(
            np.zeros([Ne, 2]), vel * np.ones([Ne, 1]), np.zeros([Ne, 1])
        )
        ego_trajs, homotopy, x0, obj_modes = TensorUtils.to_jax(
            (ego_trajs, homotopy, x0, obj_modes)
        )
        homotopy_flag = jnp.zeros([homotopy.shape[0],homotopy.shape[1],len(HomotopyType)],dtype=bool)
        if homotopy.shape[1]>0:
            for i,hc in enumerate(HomotopyType):
                homotopy_flag=homotopy_flag.at[homotopy==hc,i].set(True)
            
        

        ego_x0 = jnp.array(ego_x0)
        
        xl = jnp.concatenate([ego_x0[:, None], ego_trajs[:, :-1]], 1)
        ego_up = self.ego_dyn.inverse_dyn(xl, ego_trajs,self.dt)
        
        
        relevant_objects = {track_id.item(): self.objects[track_id.item()] for track_id in obj_idx if track_id>=0}
        if len(relevant_objects)<self.num_dynamic_object+self.num_static_object:
            pred_mode = pred_x.shape[0]
            dm = self.dummy_object
            dummy_object_tiled = self.dummy_object.copy()
            dummy_object_tiled.xp = dummy_object_tiled.xp[None,:].repeat(pred_mode,0)
            dummy_object_tiled.up = dummy_object_tiled.up[None,:].repeat(pred_mode,0)
            
            N = self.num_dynamic_object+self.num_static_object-len(relevant_objects)
            dummy_objects = {-i-1:dummy_object_tiled for i in range(N)}
            relevant_objects.update(dummy_objects)
            homotopy_flag = jnp.concatenate([homotopy_flag,jnp.zeros([homotopy_flag.shape[0],N,len(HomotopyType)],dtype=bool)],1)
        
        dyn_homotopy = homotopy_flag[:,:self.num_dynamic_object]
        static_homotopy = homotopy_flag[:,self.num_dynamic_object:]
        
        dynamic_objects = {k:v for k,v in relevant_objects.items() if k in dynamic_track_ids}

        static_objects = {k:v for k,v in relevant_objects.items() if k in static_track_ids}
            
        ego_extents = TensorUtils.to_jax(ego_extents)
        
        
        numMode = homotopy_flag.shape[0]
        

        # print("number of agents:",len(relevant_objects))
        # print("number of homotopy:",numMode)
        if ego_xref is not None:
            ego_xref = ego_xref[None,:].repeat(numMode,0)

        
            
        opt_sol = None
        current_opt = np.inf    
        feasible_idx = np.array([i for i in range(numMode)])
        T,Na,xdim,udim = self.horizon, self.num_dynamic_object,4,2
        dyn_homotopy_selected = dyn_homotopy[feasible_idx]
        static_homotopy_selected = static_homotopy[feasible_idx]
        ego_xref_selected = ego_xref[feasible_idx]
        for id, obj in dynamic_objects.items():
            obj.xi = obj.xp[obj_modes]
            obj.ui = obj.up[obj_modes]

        obj_x0 = np.stack([obj.x0 for obj in dynamic_objects.values()], 0) if len(dynamic_objects)>0 else np.zeros([0,xdim])
        obj_u0 = np.stack([obj.u0 for obj in dynamic_objects.values()], 0) if len(dynamic_objects)>0 else np.zeros([0,udim])
        max_iters_by_round = [1e3,1e2,50,30]
        obj_modes_selected = obj_modes[feasible_idx]
        # save_dict = dict(ego_x0=ego_x0, 
        #                  ego_extents=ego_extents, 
        #                  ego_trajs=ego_trajs, 
        #                  ego_up=ego_up, 
        #                  dyn_homotopy_selected=dyn_homotopy_selected, 
        #                  static_homotopy_selected=static_homotopy_selected, 
        #                  dynamic_objects=dynamic_objects,
        #                  static_objects=static_objects,
        #                  obj_modes_selected=obj_modes_selected,
        #                  lane_info=lane_info,
        #                  ego_xref_selected=ego_xref_selected,
        #                  MPC = self)
        # with open("problem.pkl","wb") as f:
        #     pickle.dump(save_dict,f)
        sol = [None]*numMode
        current_opt = np.inf
        Feasible_flag = np.ones(numMode,dtype=bool)
        for round in range(self.num_rounds):
            if self.cfg.qp_solver == "FORCESPRO":
                ego_xp,obj_xp,A,B,C,GU,LU,GC,LC,GCS,LCS,GL,LL,R,fu,Q,fx,dR = self.setup_mpc_instances(ego_x0[0], ego_extents, ego_trajs, ego_up, dyn_homotopy_selected, static_homotopy_selected, dynamic_objects,static_objects,obj_modes_selected,lane_info,ego_xref_selected)
                # Q = TensorUtils.block_diag_from_cat_jit(Q)
                # R = TensorUtils.block_diag_from_cat_jit(R)
                # dR = TensorUtils.block_diag_from_cat_jit(dR)
                
                NC = GC.shape[2]
                NCS = GCS.shape[2]
                NL = GL.shape[2]
                if self.cfg.slack_strat=="L1":
                    NS = NC+NCS+NL
                elif self.cfg.slack_strat=="Linf":
                    NS = 2
                vardim = (Na+1)*(xdim+2*udim)+NS
                Gu = np.concatenate([np.zeros([(Na+1)*udim,(Na+1)*xdim]),np.eye((Na+1)*udim),np.zeros(((Na+1)*udim,(Na+1)*udim+NS))],-1)[None].repeat(T,0)
                xinit = -np.concatenate([ego_x0[0],obj_x0.flatten(),self.u0,obj_u0.flatten()],-1)
                
                t0 = time.time()
                for i,idx in enumerate(feasible_idx):
                    problem = dict(xinit=xinit)
                    Ai = TensorUtils.block_diag_from_cat_jit(A[i])
                    Bi = TensorUtils.block_diag_from_cat_jit(B[i])
                    Gx = np.concatenate([np.concatenate([Ai,Bi,np.zeros([T,Bi.shape[1],Bi.shape[2]+NS])],-1),],-2)
                    # Gu = np.zeros([T,udim*(Na+1),vardim])
                    
                    
                    EC = np.concatenate([Gx,Gu],-2)
     
                    Ec = -C[i].reshape(T,-1)
                    Ec = np.concatenate([Ec,np.zeros([T,udim*(Na+1)])],-1)

                    if self.cfg.slack_strat == "L1":
                        gc = np.concatenate([GC[i],np.zeros([T,NC,(Na+1)*2*udim]),-np.eye(NC)[None].repeat(T,0),np.zeros([T,NC,NCS+NL])],-1)
                        gcs = np.concatenate([GCS[i],np.zeros([T,NCS,Na*xdim+(Na+1)*2*udim+NC]),-np.eye(NCS)[None].repeat(T,0),np.zeros([T,NCS,NL])],-1)
                        gl = np.concatenate([GL[i],np.zeros([T,NL,vardim-xdim-NL]),-np.eye(NL)[None].repeat(T,0)],-1)
                    elif self.cfg.slack_strat == "Linf":
                        gc = np.concatenate([GC[i],np.zeros([T,NC,(Na+1)*2*udim]),-np.ones([T,NC,1]),np.zeros([T,NC,1])],-1)
                        gcs = np.concatenate([GCS[i],np.zeros([T,NCS,Na*xdim+(Na+1)*2*udim]),-np.ones([T,NCS,1]),np.zeros([T,NCS,1])],-1)
                        gl = np.concatenate([GL[i],np.zeros([T,NL,vardim-xdim-1]),-np.ones([T,NL,1])],-1)
                    gc = np.concatenate([np.zeros_like(gc[:1]),gc],0)
                    gcs = np.concatenate([np.zeros_like(gcs[:1]),gcs],0)
                    gl = np.concatenate([np.zeros_like(gl[:1]),gl],0)
                    gu = np.concatenate([GU[i],np.zeros([T,GU.shape[2],vardim-GU.shape[-1]])],-1)
                    gu = np.concatenate([gu,np.zeros_like(gu[:1])],0)
                    IEA = np.concatenate([gc,gcs,gl,gu],-2)
                    # IEA_sparse = IEA.transpose(0,2,1).reshape(T+1,-1)[:,np.flatnonzero(self.param_sparsity["ineq.p.A"])]

                    Lx = np.concatenate([LC[i],LCS[i],LL[i]],-1)
                    Lx = np.concatenate([np.ones_like(Lx[:1]),Lx],0)
                    Lu = np.concatenate([LU[i],np.ones_like(LU[i,:1])],0)
                    IEB = np.concatenate([Lx,Lu],-1)

                    f = np.zeros([T+1,vardim])
                    f[1:,:(Na+1)*xdim] = fx[i].reshape(T,-1)
                    if self.cfg.slack_strat == "L1":
                        f[:,-NS:-NL] = self.Mcoll
                        f[:,-NL:] = self.Mlane
                    elif self.cfg.slack_strat == "Linf":
                        f[:,-2] = self.Mcoll
                        f[:,-1] = self.Mlane
                    for t in range(1,T+1):
                        problem["EC_"+str(t)] = EC[t-1]
                        problem["Ec_"+str(t+1)] = Ec[t-1]
                    for t in range(0,T+1):
                        problem["pA_"+str(t+1)] = IEA[t]
                        problem["pb_"+str(t+1)] = IEB[t]
                        problem["f_"+str(t+1)] = f[t]
                    # problem["FORCESdiagnostics"]=1
                    # sanity check
                    # D = np.concatenate([np.concatenate([-np.eye(xdim*(Na+1)),np.zeros([xdim*(Na+1),2*udim*(Na+1)+NS])],-1),
                    #     np.concatenate([np.zeros([udim*(Na+1),xdim*(Na+1)]),-np.eye((Na+1)*udim),self.dt*np.eye((Na+1)*udim),np.zeros([udim*(Na+1),NS])],-1)],-2)

                    # x0 = np.concatenate([ego_x0[0],obj_x0.flatten()],-1)
                    # xx = np.concatenate([ego_xp[i],obj_xp[i].transpose(1,0,2).reshape(T,-1)],-1)
                    # xx = np.concatenate([x0[None],xx],0)
                    
                    # obj_up = np.stack([obj.up[i] for obj in dynamic_objects.values()], 0)
                    # u0 = np.concatenate([self.u0,obj_u0.flatten()],-1)
                    # uu = np.concatenate([ego_up[i],obj_up.transpose(1,0,2).reshape(T,-1)],-1)
                    # ul = np.concatenate([u0[None],uu],0)
                    # uu = np.concatenate([uu,uu[-1:]],0)
                    # du = (uu-ul)/self.dt
                    # xu = np.concatenate([xx,uu,du,np.ones([T+1,2])*100],-1)
                    # l=(EC@xu[:T,:,None]).squeeze(-1)+(D[None]@xu[1:,:,None]).squeeze(-1)-Ec
                    
                    # ll=(IEA@xu[:,:,None]).squeeze(-1)-IEB
                    
                    # lll=(D@xu[0,...,None]).squeeze(-1)-xinit
                    
                    solve_fun = getattr(self.qp_prob, self.solvername+"_solve")
                    output,exitflag,info = solve_fun(problem)
                    if exitflag>=0:
                        sol[idx] = output.copy()
                        val = info.pobj
                        if val<current_opt:
                            opt_idx = idx
                            current_opt = val
                    else: 
                        Feasible_flag[idx] = False
                # print(f"solving time: {time.time()-t0}s")

            else:
                GC,LC,GCS,LCS,GL,LL,Ge_ego,h_ego,Ge_obj,h_obj,Gu_ego,LUE,Gu_obj,LUO,Qr_ego,Qfr_ego,Q_obj,R,fx,fu,ego_xp,obj_xp,obj_up = self.setup_mpc_instances(ego_x0[0], ego_extents, ego_trajs, ego_up, dyn_homotopy_selected, static_homotopy_selected, dynamic_objects,static_objects,obj_modes_selected,lane_info,ego_xref_selected)
            
                t0 = time.time()
                for i,idx in enumerate(feasible_idx):

                    # ECOS does not support warm start, turn it on when using OSQP or SCS
                        
                    updated_params = self.qp_prob.param_dict.keys()
                    self.qp_prob.param_dict["fu"].value = fu
                    self.qp_prob.param_dict["Geo"].value = Ge_obj[i]
                    self.qp_prob.param_dict["ho"].value = h_obj[i]
                    self.qp_prob.param_dict["Guo"].value = Gu_obj[i]
                    self.qp_prob.param_dict["Luo"].value = LUO[i]

                    self.qp_prob.param_dict["Gc"].value = GC[i]
                    self.qp_prob.param_dict["Lc"].value = LC[i]
                    self.qp_prob.param_dict["Gcs"].value = GCS[i]
                    self.qp_prob.param_dict["Lcs"].value = LCS[i]
                    self.qp_prob.param_dict["Gl"].value = GL[i]
                    self.qp_prob.param_dict["Ll"].value = LL[i]
                    self.qp_prob.param_dict["Gee"].value = Ge_ego[i]
                    self.qp_prob.param_dict["he"].value = h_ego[i]
                    self.qp_prob.param_dict["Gue"].value = Gu_ego[i]
                    self.qp_prob.param_dict["Qr_ego"].value = Qr_ego[i]
                    self.qp_prob.param_dict["Qfr_ego"].value = Qfr_ego[i]
                    self.qp_prob.param_dict["fx"].value = fx[i]
                    self.qp_prob.param_dict["Lue"].value = LUE[i]

                    # ECOS does not support warm start, turn it on when using OSQP or SCS
                    self.qp_prob.var_dict["xe"]=ego_xp[i]
                    self.qp_prob.var_dict["ue"]=ego_up[i]
                    self.qp_prob.var_dict["xo"]=obj_xp[i]
                    self.qp_prob.var_dict["uo"]=obj_up[i]
                    Cslack = self.qp_prob.var_dict["Cslack"]
                    
                    max_iters = max_iters_by_round[round] if round<len(max_iters_by_round) else max_iters_by_round[-1]
                    # val = self.qp_prob.solve(solver="OSQP", verbose=False,warm_start=True,ignore_dpp=True)
                    # add "warm_start=True" if warm start is supported
                    #TODO: get ignore_dpp from cfg
                    try:
                        if self.cfg.code_gen:
                            val = self.qp_prob.solve(method='cpg', updated_params=updated_params, verbose=False,ignore_dpp=True,max_iters=max_iters)
                        else:
                            val = self.qp_prob.solve(solver=self.cfg.qp_solver, verbose=False,warm_start=True,ignore_dpp=True,max_iters=max_iters)
                        if val==-np.inf or val==np.inf:
                            # sol[idx] = None
                            Feasible_flag[idx] = False
                        else:
                            sol[idx] = self.qp_prob.solution.copy()
                            if val<current_opt:
                                opt_idx = idx
                                current_opt = val
                    except:
                        Feasible_flag[idx] = False
                print(f"solving time: {time.time()-t0}s")
                # update initial guess
                
            feasible_idx = np.array([i for i in range(numMode) if Feasible_flag[i]])
            if feasible_idx.shape[0]==0:
                break
            if self.cfg.qp_solver=="FORCESPRO":
                ego_trajs = np.stack([sol[i]["ego_x"].reshape(T,xdim) for i in feasible_idx],0)
                obj_trajs = np.stack([sol[i]["obj_x"].reshape(T,Na,xdim) for i in feasible_idx],0).transpose(0,2,1,3) if Na>0 else None
                ego_up = np.stack([sol[i]["ego_u"].reshape(T,udim) for i in feasible_idx],0)
                obj_up = np.stack([sol[i]["obj_u"].reshape(T,Na,udim) for i in feasible_idx],0).transpose(0,2,1,3) if Na>0 else None
            else:
                xe = self.qp_prob.var_dict["xe"]
                ue = self.qp_prob.var_dict["ue"]
                uo = self.qp_prob.var_dict["uo"]
                xo = self.qp_prob.var_dict["xo"]
                ego_trajs = np.stack([sol[i].primal_vars[xe.id] for i in feasible_idx],0).reshape(-1,T,xdim)
                ego_up = np.stack([sol[i].primal_vars[ue.id] for i in feasible_idx],0).reshape(-1,T,udim)
                obj_trajs = np.stack([sol[i].primal_vars[xo.id] for i in feasible_idx],0).reshape(-1,len(dynamic_objects),T,xdim)
                obj_up = np.stack([sol[i].primal_vars[uo.id] for i in feasible_idx],0).reshape(-1,len(dynamic_objects),T,udim)
            dyn_homotopy_selected = dyn_homotopy[feasible_idx]
            static_homotopy_selected = static_homotopy[feasible_idx]
            obj_modes_selected = obj_modes[feasible_idx]
            ego_xref_selected = ego_xref[feasible_idx]
            
            for j,id in enumerate(dynamic_objects.keys()):
                dynamic_objects[id].xi = obj_trajs[:,j]
                dynamic_objects[id].ui = obj_up[:,j]

            
                    
        
        
        
        # dis0 = torch.norm(obs["agent_hist"][1:,-1,:2],dim=-1)
        # if dis0.min()<3:
        #     print("collision detected")
        # print(dis.min())
        if current_opt<np.inf:
            # if sol[opt_idx].primal_vars[Cslack.id].max()>0.3:
            #     print("collision detected")
            # all_solution = np.stack([sol for sol in opt_sol if sol is not None],0)
            if self.cfg.qp_solver=="FORCESPRO":
                xref_opt = ego_xref[opt_idx]
                ego_x = sol[opt_idx]["ego_x"].reshape(T,xdim)
                obj_x = sol[opt_idx]["obj_x"].reshape(T,Na,xdim).transpose(1,0,2) if Na>0 else None
                ego_u = sol[opt_idx]["ego_u"].reshape(T,udim)
                obj_u = sol[opt_idx]["obj_u"].reshape(T,Na,udim).transpose(1,0,2) if Na>0 else None
                if feasible_idx.shape[0]>1:
                    ego_candidate_x = np.stack([sol[i]["ego_x"].reshape(T,xdim) for i in feasible_idx if i!=opt_idx],0)
                    ego_candidate_u = np.stack([sol[i]["ego_u"].reshape(T,udim) for i in feasible_idx if i!=opt_idx],0)
                    # obj_candidate_x = np.stack([sol[i]["obj_x"].reshape(T,Na,xdim).transpose(1,0,2) for i in feasible_idx if i!=opt_idx],0)
                    # obj_candidate_u = np.stack([sol[i]["obj_u"].reshape(T,Na,udim).transpose(1,0,2) for i in feasible_idx if i!=opt_idx],0)
                else: 
                    ego_candidate_x = None
                    ego_candidate_u = None
            else:
                opt_sol = {k:sol[opt_idx].primal_vars[var.id] for k,var in self.qp_prob.var_dict.items()}
                other_sol = [solution for i,solution in enumerate(sol) if solution is not None and solution.status not in ["infeasible","unbounded"] and i!=opt_idx]
                
                if len(other_sol)>0:
                    ego_candidate_x = np.stack([x.primal_vars[xe.id] for x in other_sol],0).reshape(-1,T,xdim)
                    ego_candidate_u = np.stack([x.primal_vars[ue.id] for x in other_sol],0).reshape(-1,T,udim)
                    # obj_candidate_x = np.stack([x.primal_vars[xo.id] for x in other_sol],0).reshape(-1,dynamic_track_ids.shape[0],T,xdim)
                    # obj_candidate_u = np.stack([x.primal_vars[uo.id] for x in other_sol],0).reshape(-1,dynamic_track_ids.shape[0],T,udim)
                else:
                    ego_candidate_x = None
                    ego_candidate_u = None
                
                
                xref_opt = ego_xref[opt_idx]
                ego_x = opt_sol["xe"].reshape(T,xdim)
                obj_x = opt_sol["xo"].reshape(-1,T,xdim)
                ego_u = opt_sol["ue"].reshape(T,udim)
                obj_u = opt_sol["uo"].reshape(-1,T,udim)

            
            # debug
            
            # xev = opt_sol["xe"]
            # xe1 = xev.reshape(T,xdim)

            # if np.abs(xe1[:,-1]).max()>0.5:
            #     print("weird")

            # uev = opt_sol["ue"]
            # ue1 = uev.reshape(T,udim)
            # xov = opt_sol["xo"]
            # xo1 = xov.reshape(dynamic_track_ids.shape[0],T,xdim)
            # uov = opt_sol["uo"]
            # uo1 = uov.reshape(dynamic_track_ids.shape[0],T,udim)
            # xx1 = np.hstack([xev,xov])
            # uu1 = np.hstack([uev,uov])
            # xu1 = np.hstack([xx1,uu1])

            # xe0 = ego_xref[opt_idx]
            # ue0 = ego_up[opt_idx]
            # xo0 = obj_xp[opt_idx]
            # uo0 = obj_up[opt_idx]

            # xx0 = np.hstack([xe0.flatten(),xo0.flatten()])
            # uu0 = np.hstack([ue0.flatten(),uo0.flatten()])
            # xu0 = np.hstack([xx0,uu0])

            # i = opt_idx
            # fu = fu
            # Geo = Ge_obj[i]
            # ho = h_obj[i]
            # ho = Gu_obj[i]
            # Luo = LUO[i]

            # Gc = GC[i]
            # Lc = LC[i]
            # Gcs = GCS[i]
            # Lcs = LCS[i]
            # Gl = GL[i]
            # Ll = LL[i]
            # Gee = Ge_ego[i]
            # he = h_ego[i]
            # Gue = Gu_ego[i]
            # Qr_ego1 = Qr_ego[i]
            # Qfr_ego1 = Qfr_ego[i]
            # fx1 = fx[i]
            # Lue = LUE[i]

            # obj1=(1/2)*(xx1.T@Q_obj@xx1+xev.T@Qr_ego[opt_idx].T@Qr_ego[opt_idx]@xev+uu1.T@R@uu1) + fx[opt_idx].T@xx1 + fu.T@uu1
            # obj0=(1/2)*(xx0.T@Q_obj@xx0+xe0.flatten().T@Qr_ego[opt_idx].T@Qr_ego[opt_idx]@xe0.flatten()+uu0.T@R@uu0) + fx[opt_idx].T@xx0 + fu.T@uu0

            # xx1.T@Q_obj@xx1+fx[opt_idx,64:].T@xo1.flatten()
            # xx0.T@Q_obj@xx0+fx[opt_idx,64:].T@xo0.flatten()

            # j=0
            # homotopy = homotopy_flag[opt_idx,j]
 
            
        else:
            ego_x = ego_trajs[0]
            ego_u = ego_up[0]
            obj_x = np.zeros([0,T,xdim])
            obj_u = np.zeros([0,T,udim])
            xref_opt = None
            ego_candidate_x = None
            ego_candidate_u = None

        

            save_dict = dict(
                x0=x0,
                dyn_homotopy=dyn_homotopy,
                static_homotopy = static_homotopy,
                dynamic_objects=dynamic_objects,
                static_objects = static_objects,
                obj_modes=obj_modes,
                ego_xp=ego_trajs,
                ego_up=ego_up,
                ego_x0=ego_x0,
                lane_info=lane_info,
            )
            inputs = dict(obs=obs,ego_traj=ego_trajs_full,pred_x=pred_x,ego_trajs=ego_trajs)
            with open("obs.pkl", "wb") as f:
                pickle.dump(inputs, f)

            with open("problem.pkl", "wb") as f:
                pickle.dump(save_dict, f)
        
        simt_out = np.insert(simt,0,0)
        mpct_u = self.dt*np.arange(0,self.horizon)
        f = interp1d(mpct,ego_x,axis=0,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
        ego_x = f(simt)
        f = interp1d(mpct_u,ego_u,axis=0,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
        ego_u = f(simt_out)
        self.u0 = ego_u[1]
        if obj_x is not None:
            f = interp1d(mpct,obj_x,axis=1,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            obj_x = f(simt)
            f = interp1d(mpct_u,obj_u,axis=1,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            obj_u = f(simt_out)
        if ego_candidate_x is not None:
            f = interp1d(mpct,ego_candidate_x,axis=1,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            ego_candidate_x = f(simt)
            f = interp1d(mpct_u,ego_candidate_u,axis=1,assume_sorted=True,bounds_error=False,fill_value="extrapolate")
            ego_candidate_u = f(simt_out)

        if False:
            # change from xyvh to xyhv
            ego_x = np.concatenate([ego_x[:,:2],ego_x[:,3:4],ego_x[:,2:3]],-1)
            obj_x = np.concatenate([obj_x[:,:,:2],obj_x[:,:,3:4],obj_x[:,:,2:3]],-1)
            if ego_candidate_x is not None:
                ego_candidate_x = np.concatenate([ego_candidate_x[:,:,:2],ego_candidate_x[:,:,3:4],ego_candidate_x[:,:,2:3]],-1)
            if xref_opt is not None:
                xref_opt = np.concatenate([xref_opt[:,:2],xref_opt[:,3:4],xref_opt[:,2:3]],-1)
        return dict(ego_x=ego_x, ego_u = ego_u, obj_x=obj_x, obj_u = obj_u, homotopy = homotopy,track_ids=obj_idx,xref=xref_opt,ego_candidate_x=ego_candidate_x,ego_candidate_u=ego_candidate_u)
class SQPMPCController(Policy):
    def __init__(self, device, cfg, predictor, sampler, savetrace=True, *args, **kwargs):
        super().__init__(device, *args, **kwargs)
        self.device = device
        self.predictor = predictor
        self.sampler = sampler
        self.cfg = cfg
        self.dt = cfg.dt
        self.pred_dt = cfg.pred_dt
        self.horizon_sec = cfg.horizon_sec
        self.horizon = int(cfg.horizon_sec / self.dt)
        self.solver = dict()
        self.timer = Timers()
        self.time_stamp = 0
        self.lane_change_interval = cfg.lane_change_interval if "lane_change_interval" in cfg else None
        cache_path = Path("~/.unified_data_cache").expanduser()
        self.mapAPI = MapAPI(cache_path)
        self.vec_map = dict()
        self.num_dynamic_object = cfg.num_dynamic_object #fix number of agents to avoid jit recompilation
        self.num_static_object = cfg.num_static_object
        self.savetrace = savetrace
        if savetrace:
            self.trace = dict()
        self.desired_lane = defaultdict(lambda: None)

    def reset(self):
        self.solver.clear()
        self.trace.clear()
        self.desired_lane = defaultdict(lambda: None)

    def eval(self):
        self.predictor.eval()

    @staticmethod
    def get_scene_obs(ego_obs, agent_obs, goal=None, max_num_agents=None):
        # turn the observations into scene-centric form

        centered_raster_from_agent = ego_obs["raster_from_agent"][0]
        centered_agent_from_raster, _ = torch.linalg.inv_ex(centered_raster_from_agent)
        num_agents = [
            sum(agent_obs["scene_index"] == i) + 1 if len(agent_obs["scene_index"])>0 else 1 for i in ego_obs["scene_index"]
        ] 
        num_agents = torch.tensor(num_agents, device=ego_obs["scene_index"].device)
        hist_pos_b = list()
        hist_yaw_b = list()
        hist_avail_b = list()
        fut_pos_b = list()
        fut_yaw_b = list()
        fut_avail_b = list()
        agent_from_center_b = list()
        center_from_agent_b = list()
        # raster_from_center_b = list()
        center_from_raster_b = list()
        # raster_from_world_b = list()
        maps_b = list()
        curr_speed_b = list()
        type_b = list()
        extents_b = list()
        scene_goal_b = list()
        track_id_b = list()

        device = ego_obs["agent_hist"].device

        for i, scene_idx in enumerate(ego_obs["scene_index"]):
            if num_agents[i]>1:
                agent_idx = torch.where(agent_obs["scene_index"] == scene_idx)[0]
                if goal is not None:
                    scene_goal_b.append(
                        torch.cat((torch.zeros_like(goal[0:1]), goal[agent_idx]), 0)
                    )
                center_from_agent = (
                    ego_obs["agent_from_world"][i].unsqueeze(0)
                    @ agent_obs["world_from_agent"][agent_idx]
                )
                center_from_agents = torch.cat(
                    (
                        torch.eye(3, device=center_from_agent.device).unsqueeze(0),
                        center_from_agent,
                    ),
                    0,
                )

                hist_pos_raw = torch.cat(
                    (
                        ego_obs["history_positions"][i : i + 1],
                        agent_obs["history_positions"][agent_idx],
                    ),
                    0,
                )
                hist_yaw_raw = torch.cat(
                    (
                        ego_obs["history_yaws"][i : i + 1],
                        agent_obs["history_yaws"][agent_idx],
                    ),
                    0,
                )

                agents_hist_avail = torch.cat(
                    (
                        ego_obs["history_availabilities"][i : i + 1],
                        agent_obs["history_availabilities"][agent_idx],
                    ),
                    0,
                )
                agents_hist_pos = GeoUtils.transform_points_tensor(
                    hist_pos_raw, center_from_agents
                ) * agents_hist_avail.unsqueeze(-1)
                agents_hist_yaw = (
                    hist_yaw_raw
                    + torch.cat(
                        (ego_obs["yaw"][i : i + 1], agent_obs["yaw"][agent_idx]), 0
                    )[:, None, None]
                    - ego_obs["yaw"][i]
                ) * agents_hist_avail.unsqueeze(-1)

                hist_pos_b.append(agents_hist_pos)
                hist_yaw_b.append(agents_hist_yaw)
                hist_avail_b.append(agents_hist_avail)
                if (
                    agent_obs["target_availabilities"].shape[1]
                    < ego_obs["target_availabilities"].shape[1]
                ):
                    pad_shape = (
                        agent_obs["target_availabilities"].shape[0],
                        ego_obs["target_availabilities"].shape[1]
                        - agent_obs["target_availabilities"].shape[1],
                    )
                    agent_obs["target_availabilities"] = torch.cat(
                        (
                            agent_obs["target_availabilities"],
                            torch.zeros(
                                pad_shape, device=agent_obs["target_availabilities"].device
                            ),
                        ),
                        1,
                    )
                agents_fut_avail = torch.cat(
                    (
                        ego_obs["target_availabilities"][i : i + 1],
                        agent_obs["target_availabilities"][agent_idx],
                    ),
                    0,
                )
                if (
                    agent_obs["target_positions"].shape[1]
                    < ego_obs["target_positions"].shape[1]
                ):
                    pad_shape = (
                        agent_obs["target_positions"].shape[0],
                        ego_obs["target_positions"].shape[1]
                        - agent_obs["target_positions"].shape[1],
                        *agent_obs["target_positions"].shape[2:],
                    )
                    agent_obs["target_positions"] = torch.cat(
                        (
                            agent_obs["target_positions"],
                            torch.zeros(
                                pad_shape, device=agent_obs["target_positions"].device
                            ),
                        ),
                        1,
                    )
                    pad_shape = (
                        agent_obs["target_yaws"].shape[0],
                        ego_obs["target_yaws"].shape[1] - agent_obs["target_yaws"].shape[1],
                        *agent_obs["target_yaws"].shape[2:],
                    )
                    agent_obs["target_yaws"] = torch.cat(
                        (
                            agent_obs["target_yaws"],
                            torch.zeros(pad_shape, device=agent_obs["target_yaws"].device),
                        ),
                        1,
                    )
                fut_pos_raw = torch.cat(
                    (
                        ego_obs["target_positions"][i : i + 1],
                        agent_obs["target_positions"][agent_idx],
                    ),
                    0,
                )
                fut_yaw_raw = torch.cat(
                    (
                        ego_obs["target_yaws"][i : i + 1],
                        agent_obs["target_yaws"][agent_idx],
                    ),
                    0,
                )
                agents_fut_pos = GeoUtils.transform_points_tensor(
                    fut_pos_raw, center_from_agents
                ) * agents_fut_avail.unsqueeze(-1)
                agents_fut_yaw = (
                    fut_yaw_raw
                    + torch.cat(
                        (ego_obs["yaw"][i : i + 1], agent_obs["yaw"][agent_idx]), 0
                    )[:, None, None]
                    - ego_obs["yaw"][i]
                ) * agents_fut_avail.unsqueeze(-1)
                fut_pos_b.append(agents_fut_pos)
                fut_yaw_b.append(agents_fut_yaw)
                fut_avail_b.append(agents_fut_avail)

                track_id_b.append(
                    torch.cat(
                        (
                            ego_obs["track_id"][i].unsqueeze(0),
                            agent_obs["track_id"][agent_idx],
                        )
                    )
                )

                curr_yaw = agents_hist_yaw[:, -1]
                curr_pos = agents_hist_pos[:, -1]
                agents_from_center = GeoUtils.transform_matrices(
                    -curr_yaw.flatten(), torch.zeros_like(curr_pos)
                ) @ GeoUtils.transform_matrices(
                    torch.zeros_like(curr_yaw).flatten(), -curr_pos
                )

                # raster_from_center = centered_raster_from_agent @ agents_from_center
                center_from_raster = center_from_agents @ centered_agent_from_raster

                # raster_from_world = torch.cat((ego_obs["raster_from_world"][i:i+1],agent_obs["raster_from_world"][agent_idx]),0)

                agent_from_center_b.append(agents_from_center)
                center_from_agent_b.append(center_from_agents)
                # raster_from_center_b.append(raster_from_center)
                center_from_raster_b.append(center_from_raster)
                # raster_from_world_b.append(raster_from_world)

                maps = torch.cat(
                    (ego_obs["image"][i : i + 1], agent_obs["image"][agent_idx]), 0
                )
                curr_speed = torch.cat(
                    (ego_obs["curr_speed"][i : i + 1], agent_obs["curr_speed"][agent_idx]),
                    0,
                )
                agents_type = torch.cat(
                    (ego_obs["type"][i : i + 1], agent_obs["type"][agent_idx]), 0
                )
                agents_extent = torch.cat(
                    (ego_obs["extent"][i : i + 1], agent_obs["extent"][agent_idx]), 0
                )
            else:
                if goal is not None:
                    scene_goal_b.append(torch.zeros_like(goal[0:1]))
                center_from_agents = torch.eye(3,device=device).unsqueeze(0)


                hist_pos_raw = ego_obs["history_positions"][i : i + 1]
                hist_yaw_raw = ego_obs["history_yaws"][i : i + 1]

                agents_hist_avail = ego_obs["history_availabilities"][i : i + 1]



                hist_pos_b.append(hist_pos_raw)
                hist_yaw_b.append(hist_yaw_raw)
                hist_avail_b.append(agents_hist_avail)
                
                agents_fut_avail = ego_obs["target_availabilities"][i : i + 1]
                fut_pos_raw = ego_obs["target_positions"][i : i + 1]
                fut_yaw_raw = ego_obs["target_yaws"][i : i + 1]

                agents_fut_yaw = ego_obs["target_yaws"][i : i + 1]
                fut_pos_b.append(fut_pos_raw)
                fut_yaw_b.append(fut_yaw_raw)
                fut_avail_b.append(agents_fut_avail)

                track_id_b.append(ego_obs["track_id"][i].unsqueeze(0))

                curr_yaw = hist_yaw_raw[:, -1]
                curr_pos = hist_pos_raw[:, -1]
                agents_from_center = torch.eye(3,device=device).unsqueeze(0)

                # raster_from_center = centered_raster_from_agent @ agents_from_center
                center_from_raster = center_from_agents @ centered_agent_from_raster

                # raster_from_world = torch.cat((ego_obs["raster_from_world"][i:i+1],agent_obs["raster_from_world"][agent_idx]),0)

                agent_from_center_b.append(agents_from_center)
                center_from_agent_b.append(center_from_agents)
                # raster_from_center_b.append(raster_from_center)
                center_from_raster_b.append(center_from_raster)
                # raster_from_world_b.append(raster_from_world)

                maps = ego_obs["image"][i : i + 1]
                curr_speed = ego_obs["curr_speed"][i : i + 1]
                agents_type = ego_obs["type"][i : i + 1]
                agents_extent = ego_obs["extent"][i : i + 1]
            maps_b.append(maps)
            curr_speed_b.append(curr_speed)
            type_b.append(agents_type)
            extents_b.append(agents_extent)
        if goal is not None:
            scene_goal = pad_sequence(scene_goal_b, batch_first=True, padding_value=0)
        else:
            scene_goal = None

        history_positions=pad_sequence(
                hist_pos_b, batch_first=True, padding_value=0
            )
        history_yaws=pad_sequence(hist_yaw_b, batch_first=True, padding_value=0)
        history_availabilities=pad_sequence(
                hist_avail_b, batch_first=True, padding_value=0
            )
        history_pos_world = GeoUtils.batch_nd_transform_points(history_positions,ego_obs["world_from_agent"][:,None,None])*history_availabilities.unsqueeze(-1)
        history_yaws_world = GeoUtils.round_2pi(history_yaws + ego_obs["yaw"][:,None,None,None])
        
        scene_obs = dict(
            num_agents=num_agents,
            image=pad_sequence(maps_b, batch_first=True, padding_value=0),
            target_positions=pad_sequence(fut_pos_b, batch_first=True, padding_value=0),
            target_yaws=pad_sequence(fut_yaw_b, batch_first=True, padding_value=0),
            target_availabilities=pad_sequence(
                fut_avail_b, batch_first=True, padding_value=0
            ),
            history_positions=history_positions,
            history_yaws=history_yaws,
            history_availabilities=history_availabilities,
            history_pos_world = history_pos_world,
            history_yaws_world = history_yaws_world,
            curr_speed=pad_sequence(curr_speed_b, batch_first=True, padding_value=0),
            centroid=ego_obs["centroid"],
            yaw=ego_obs["yaw"],
            agent_type=pad_sequence(type_b, batch_first=True, padding_value=0),
            extent=pad_sequence(extents_b, batch_first=True, padding_value=0),
            raster_from_agent=ego_obs["raster_from_agent"],
            agent_from_raster=centered_agent_from_raster,
            # raster_from_center=pad_sequence(raster_from_center_b,batch_first=True,padding_value=0),
            # center_from_raster=pad_sequence(center_from_raster_b,batch_first=True,padding_value=0),
            agents_from_center=pad_sequence(
                agent_from_center_b, batch_first=True, padding_value=0
            ),
            center_from_agents=pad_sequence(
                center_from_agent_b, batch_first=True, padding_value=0
            ),
            # raster_from_world=pad_sequence(raster_from_world_b,batch_first=True,padding_value=0),
            agent_from_world=ego_obs["agent_from_world"],
            world_from_agent=ego_obs["world_from_agent"],
            track_id=pad_sequence(track_id_b, batch_first=True, padding_value=-1),
            scene_index=ego_obs["scene_index"],
            dt=ego_obs["dt"],
        )
        

        if (
            max_num_agents is not None
            and scene_obs["num_agents"].max() > max_num_agents
        ):
            dis = torch.norm(scene_obs["history_positions"][:, :, -1], dim=-1)
            dis = dis.masked_fill_(
                ~scene_obs["history_availabilities"][..., -1], np.inf
            )
            idx = torch.argsort(dis, dim=1)[:, :max_num_agents]
            for k, v in scene_obs.items():
                if v.shape[:2] == dis.shape:
                    scene_obs[k] = TensorUtils.gather_from_start_single(
                        scene_obs[k], idx
                    )
            if scene_goal is not None:
                scene_goal = TensorUtils.gather_from_start_single(scene_goal, idx)

        return scene_obs, scene_goal

    @staticmethod
    def extend_prediction(pred_traj, desired_horizon, dt):
        assert desired_horizon > pred_traj.shape[-2]
        dT = desired_horizon - pred_traj.shape[-2]
        x0 = pred_traj[..., -1, :]
        xp = SQPMPCController.constvelpred(x0, dT, dt)
        return torch.cat((pred_traj, xp), -2)

    def get_prediction_sc(self, scene_obs):
        """_summary_

        Args:
            scene_obs (dict): scene centric observation

        Returns:
            pred_traj: [b,M,Na,T+1,4]
            pred_u: [b,M,Na,T,2]
        """
        dt = scene_obs["dt"][0]
        bs = scene_obs["num_agents"].shape[0]
        Na = scene_obs["num_agents"].max()
        pos0 = scene_obs["history_positions"][..., -1, :]
        yaw0 = scene_obs["history_yaws"][..., -1, :]
        vel0 = scene_obs["curr_speed"].unsqueeze(-1)

        if self.predictor is not None:
            self.timer.tic("prediction")
            agent_pred = self.predictor.predict(scene_obs)
            numMode = agent_pred["trajectories"].shape[2]
            self.timer.toc("prediction")
            pos_pred = agent_pred["trajectories"][..., :2]
            yaw_pred = agent_pred["trajectories"][..., 2:]
            pred_traj = Unicycle.get_state(
                pos_pred,
                yaw_pred,
                dt,
                mask=torch.ones_like(pos_pred[..., 0], dtype=torch.bool),
            )
            if pred_traj.ndim==6:
                pred_traj=pred_traj[:,0]
            pred_traj = pred_traj.reshape(bs, numMode, Na, -1, 4)  # x,y,v,yaw

        else:
            pred_traj = self.constvelpred(pos0, vel0, yaw0, int(self.horizon_sec/sim_dt), dt).unsqueeze(1)
        if pred_traj.shape[-2] < int(self.horizon_sec/sim_dt):
            pred_traj = self.extend_prediction(pred_traj, int(self.horizon_sec/sim_dt), dt)
        # TODO: include other dynamic types

        x0 = Unicycle.combine_to_state(pos0, vel0, yaw0)
        xhist = Unicycle.get_state(
            scene_obs["history_positions"],
            scene_obs["history_yaws"],
            dt,
            scene_obs["history_availabilities"],
        )
        u0 = Unicycle.inverse_dyn(xhist[..., -2, :], xhist[..., -1, :], dt)
        pred_traj = torch.cat((x0.unsqueeze(-2).unsqueeze(1).repeat_interleave(numMode,1), pred_traj), -2)
        pred_u = Unicycle.inverse_dyn(pred_traj[..., :-1, :], pred_traj[..., 1:, :], dt)
        pred_u = torch.cat((u0.unsqueeze(-2).unsqueeze(1).repeat_interleave(numMode,1), pred_u), -2)

        return pred_traj, pred_u

    @staticmethod
    def constvelpred(x0, horizon, dt):
        # assumes a unicycle model for all agents
        pos = Unicycle.state2pos(x0)
        yaw = Unicycle.state2yaw(x0)
        vel = Unicycle.state2vel(x0)

        s = vel * torch.arange(1, horizon + 1, device=pos.device) * dt
        xyp = torch.stack((s * torch.cos(yaw), s * torch.sin(yaw)), -1)
        xyp = xyp + pos.unsqueeze(-2)

        return Unicycle.combine_to_state(
            xyp,
            vel.unsqueeze(-2).repeat_interleave(horizon, -2),
            yaw.unsqueeze(-2).repeat_interleave(horizon, -2),
        )


        



    def obtain_lane_info(self,obs,desired_lane=None):
        device = obs["history_pos_world"].device
        ego_xyz = TensorUtils.to_numpy(torch.cat([obs["history_pos_world"][0,-1],torch.zeros(1,device=device)],-1))
        ego_xyh = TensorUtils.to_numpy(torch.cat([obs["history_pos_world"][0,-1],obs["history_yaws_world"][0,-1]],-1))
        vec_map = self.vec_map[obs["map_names"]]
        ego_lane = None
        yaw = ego_xyh[-1]
        close_lanes=vec_map.get_lanes_within(ego_xyz,10)
        dis = list()
        if len(close_lanes)>0:
            opt_dis = np.inf
            for lane in close_lanes:
                if np.abs(GeoUtils.round_2pi(np.abs(lane.center.h-yaw))).mean()>np.pi/2:
                    dis.append(np.inf)
                    continue
                dx,dy,dh = GeoUtils.batch_proj(ego_xyh,lane.center.points[:,[0,1,3]])
                idx = np.abs(dx).argmin()
                dis_i = np.abs(dy[idx])+np.abs(dh)*2
                dis.append(dis_i.item())
            ego_lane = close_lanes[np.argmin(dis)]
        if ego_lane is None:
            ego_lane = vec_map.get_closest_lane(ego_xyz)
        if desired_lane is None:
            desired_lane = ego_lane
        else:
            if desired_lane not in close_lanes or dis[close_lanes.index(desired_lane)]>5:
                desired_lane = ego_lane
        left_lane = ego_lane
        right_lane = ego_lane
        while len(left_lane.adj_lanes_left)>0:
            left_lane = vec_map.get_road_lane(list(left_lane.adj_lanes_left)[0])
        while len(right_lane.adj_lanes_right)>0:
            right_lane = vec_map.get_road_lane(list(right_lane.adj_lanes_right)[0])
        

        if len(desired_lane.next_lanes)>0:
            desired_lane_next = vec_map.get_road_lane(list(desired_lane.next_lanes)[0])
            C_xy,C_h = LaneUtils.get_bdry_xyh(desired_lane,desired_lane_next,dir="C")
        else:
            C_xy,C_h = LaneUtils.get_bdry_xyh(desired_lane,dir="C")
        if len(left_lane.next_lanes)>0:
            left_lane_next = vec_map.get_road_lane(list(left_lane.next_lanes)[0])

            LB_xy,LB_h = LaneUtils.get_bdry_xyh(left_lane,left_lane_next,dir="L")
        else:
            LB_xy,LB_h = LaneUtils.get_bdry_xyh(left_lane,dir="L")
        if len(right_lane.next_lanes)>0:
            right_lane_next = vec_map.get_road_lane(list(right_lane.next_lanes)[0])

            RB_xy,RB_h = LaneUtils.get_bdry_xyh(right_lane,right_lane_next,dir="R")
        else:
            RB_xy,RB_h = LaneUtils.get_bdry_xyh(right_lane,dir="R")

        agent_from_world = TensorUtils.to_numpy(obs["agent_from_world"])
        yaw = TensorUtils.to_numpy(obs["yaw"])
        C_xy = GeoUtils.batch_nd_transform_points_np(C_xy,agent_from_world)
        LB_xy = GeoUtils.batch_nd_transform_points_np(LB_xy,agent_from_world)
        RB_xy = GeoUtils.batch_nd_transform_points_np(RB_xy,agent_from_world)
        C_h = GeoUtils.round_2pi(C_h-yaw)
        LB_h = GeoUtils.round_2pi(LB_h-yaw)
        RB_h = GeoUtils.round_2pi(RB_h-yaw)
        center = np.concatenate([C_xy,C_h[:,None]],-1)
        delta_x,delta_y,_ = GeoUtils.batch_proj(np.array([0,0,0]),center)
        idx = np.abs(delta_x).argmin()
        leftbdry = np.concatenate([LB_xy,LB_h[:,None]],-1)
        rightbdry = np.concatenate([RB_xy,RB_h[:,None]],-1)
        lane_info = dict(center=center,leftbdry=leftbdry,rightbdry=rightbdry)
        
            
        return lane_info,desired_lane


    def get_action(self, obs_dict, **kwargs):
        self.time_stamp+=1
        for map_name in obs_dict["map_names"]:
            if map_name not in self.vec_map:
                self.vec_map[map_name] = self.mapAPI.get_map(map_name, scene_cache=None)

        print(obs_dict["scene_ts"][0].item())
            
        assert "agent_obs" in kwargs
        agent_obs = kwargs["agent_obs"]
        scene_obs, _ = self.get_scene_obs(obs_dict, agent_obs)
        device = scene_obs["history_positions"].device
        agent_pred_x, agent_pred_u = self.get_prediction_sc(scene_obs)
        bs = len(obs_dict["scene_ids"])
        ego_plan = list()
        obj_plan = list()
        
        max_numagent = scene_obs["history_positions"].shape[1]
        agents_k = [k for k,v in scene_obs.items() if isinstance(v, torch.Tensor) and v.ndim>=2 and v.shape[1] == max_numagent]
        obs_common = {k: v for k, v in scene_obs.items() if not isinstance(v,torch.Tensor) or v.shape[0]!=bs}
        for i, scene_id in enumerate(obs_dict["scene_ids"]):
            if scene_id not in self.solver:
                self.solver[scene_id] = SQPMPC(self.cfg, self.sampler, self.device)
                
            obs_i = {
                k: v[i] for k, v in scene_obs.items() if isinstance(v,torch.Tensor) and v.shape[0]==bs
            }
            obs_i.update(obs_common)
            obs_i["map_names"] = obs_dict["map_names"][i]
            num_agents = obs_i["num_agents"].item()
            pred_x_i = agent_pred_x[i,:, 1:num_agents]
            pred_u_i = agent_pred_u[i,:, 1:num_agents]
            if self.lane_change_interval is not None and (self.time_stamp+1)%self.lane_change_interval==0:
                lane_change = True
                traj_sampler = self.solver[scene_id].ego_sampler
                
            else:
                lane_change = False
                traj_sampler = None
            vel = TensorUtils.to_numpy(obs_i["curr_speed"][0])
            if self.desired_lane[scene_id] is not None:
                xyh = TensorUtils.to_numpy(torch.cat([obs_i["history_pos_world"][0,-1],obs_i["history_yaws_world"][0,-1]],-1))
                delta_x,delta_y,_ = GeoUtils.batch_proj(xyh,self.desired_lane[scene_id].center.points[:,[0,1,3]])
                idx = np.argmin(np.abs(delta_x))
                
                if np.abs(delta_y[idx])>8.0:
                    # deviate from the desired lane
                    self.desired_lane[scene_id] = None
                elif max(-delta_x)-vel*self.horizon_sec<5.0:
                    if len(self.desired_lane[scene_id].next_lanes)>0:
                        self.desired_lane[scene_id] = self.vec_map[obs_i["map_names"]].get_road_lane(list(self.desired_lane[scene_id].next_lanes)[0])

            if lane_change:
                ego_xyz = TensorUtils.to_numpy(torch.cat([obs_i["history_pos_world"][0,-1],torch.zeros(1,device=device)],-1))
                # ego_xyh = TensorUtils.to_numpy(torch.cat([world_pos,obs["world_yaw"].unsqueeze(-1)],-1))
                # yaw = TensorUtils.to_numpy(obs["world_yaw"])
                candidates=self.vec_map[obs_i["map_names"]].get_lanes_within(ego_xyz,5)
                traj0 = torch.tensor([0.0, 0.0, vel.item(), 0, 0.0, 0.0, 0.0],device=device)
                Tp = pred_x_i.shape[-2]-1

                if len(candidates)>1:
                    min_clearance = list()
                    agent_from_world = obs_i["agent_from_world"].type(torch.float32)
                    for j,lane in enumerate(candidates):
                        lane_xyh = TensorUtils.to_torch(lane.center.xyh,device=device).type(torch.float32)
                        lane_xyh[...,:2] = GeoUtils.batch_nd_transform_points(lane_xyh[...,:2],agent_from_world)
                        lane_xyh[...,2]-=obs_i["yaw"]
                        
                        sample_traj,_ = traj_sampler.gen_trajectories(traj0, self.pred_dt*Tp, lanes=lane_xyh[None], dyn_filter=True, N=Tp+1,lane_only=True)
                        sample_traj=sample_traj[...,TRAJ_INDEX]
                        if sample_traj.shape[0]>0:
                            clearance = torch.norm(sample_traj[:,None,:,:2]-pred_x_i[:1,:,1:,:2],dim=-1)
                            min_clearance.append(clearance.min(2)[0].min(1)[0].max().item())
                        else:
                            min_clearance.append(np.pi)
                    self.desired_lane[scene_id] = candidates[np.argmax(min_clearance)]
                    
                        
                
            lane_info,desired_lane = self.obtain_lane_info(obs_i,self.desired_lane[scene_id])
            self.desired_lane[scene_id] = desired_lane
            
            
            
            obs_i_active = {k:v[:num_agents] for k,v in obs_i.items() if k in agents_k}
            obs_i_active.update({k:v for k,v in obs_i.items() if k not in agents_k})

            mpc_sol = self.solver[scene_id].solve_mpc(self.time_stamp, obs_i_active, pred_x_i, pred_u_i,lane_info)
            del lane_info
            mpc_sol = TensorUtils.to_numpy(mpc_sol)
            if self.savetrace:
                if scene_id not in self.trace:
                    self.trace[scene_id] =dict()
                self.trace[scene_id][obs_dict["scene_ts"][i].item()] = mpc_sol
            ego_plan.append(mpc_sol["ego_x"])
            obj_plan.append(mpc_sol["obj_x"])

        ego_plan = torch.stack(TensorUtils.to_torch(ego_plan,device=device), 0)
        # obj_plan=pad_sequence(
        #         TensorUtils.to_torch(obj_plan,device=device), batch_first=True, padding_value=0
        #     )
        
        action = Action(positions=ego_plan[..., :2], yaws=ego_plan[..., 3:])
        del scene_obs,agent_pred_x,agent_pred_u

        return action, {}


def traverse_pick_first_ego_xp(tree:HomotopyTree,result):
    result[tree] = tree.ego_xp[0]
    return result

def traverse_obtain_homotopy(tree:HomotopyTree,result):
    result[tree] = tree.homotopy
    return result


def test():
    from tbsim.configs.algo_config import SQPMPCConfig

    cfg = SQPMPCConfig()
    cfg.dt = 0.5
    device = "cuda"
    horizon = int(cfg.horizon_sec / cfg.dt)
    ego_sampler = SplinePlanner(device, N_seg=horizon + 1,vbound=[-10,50.0])
    mpc = SQPMPC(cfg, ego_sampler, device,qp_solver_dir="tbsim/policies/MPC/qp_solver")
    import pickle

    with open("problem.pkl", "rb") as f:
        data = pickle.load(f)
    ego_extent = np.array([4.0, 2.2, 1.8])
    
    # ego_xp_dict = OrderedDict()
    # ego_xp_dict,_ = depth_first_traverse(data["homotopy_tree"],traverse_pick_first_ego_xp, {}, ego_xp_dict)
    # homotopy_dict = OrderedDict()
    # homotopy_dict,_ = depth_first_traverse(data["homotopy_tree"],traverse_obtain_homotopy, {}, homotopy_dict)
    # homotopy_list = jnp.stack(list(homotopy_dict.values()),0)
    # ego_xp = jnp.stack(list(ego_xp_dict.values()),0)
    ego_x0 = data["ego_x0"]
    ego_xp = data["ego_xp"]
    xl = jnp.concatenate([ego_x0[:, None], ego_xp[:, :-1]], 1)
    ego_up = Unicycle.inverse_dyn(xl, ego_xp, mpc.dt)
    for i in range(100):
        t1 = time.time()
        mpc.setup_mpc_instances(
            data["ego_x0"][0],
            ego_extent,
            ego_xp,
            ego_up,
            data["dyn_homotopy"],
            data["static_homotopy"],
            data["dynamic_objects"],
            data["static_objects"],
            data["obj_modes"],
            data["lane_info"],
            xref = np.zeros([5,6,4])
        )
        runtime = time.time()-t1
        print(f"runtime is {runtime}s")


def plotting():
    import pickle

    with open("obs.pkl", "rb") as f:
        data = pickle.load(f)
    obs = data["obs"]
    obs = TensorUtils.to_numpy(obs)
    ego_trajs = data["ego_trajs"]
    agent_pred = TensorUtils.to_numpy(data["pred_x"])
    trans_mat = obs["raster_from_agent"]
    img = obs["image"][0,-3:].transpose(1,2,0)
    img = (img+1)/2
    img = VisUtils.draw_agent_boxes(
        img,
        pos=obs["history_positions"][:,-1],
        yaw=obs["history_yaws"][:,-1],
        extent=obs["extent"][:,:2],
        raster_from_agent=obs["raster_from_agent"],
        outline_color=VisUtils.COLORS["ego_contour"],
        fill_color=VisUtils.COLORS["ego_fill"]
    )


    from PIL import Image, ImageDraw
    plan_marker_size=2
    im = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)
    pos_raster = GeoUtils.batch_nd_transform_points_np(
            ego_trajs[...,:2], trans_mat[None,None,:])
    pos_raster = pos_raster.reshape(-1,2)
    for pos in pos_raster:
        circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
        draw.ellipse(circle.astype(int).tolist(), fill=(0,0,255))

    pred_raster = GeoUtils.batch_nd_transform_points_np(
            agent_pred[...,:2], trans_mat[None,None,:])
    pred_raster = pred_raster.reshape(-1,2)
    for pos in pred_raster:
        circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
        draw.ellipse(circle.astype(int).tolist(), fill=(0,128,0))
    im.save("plot.jpeg")

if __name__ == "__main__":
    test()

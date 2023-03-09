from Pplan.Sampling.spline_planner import SplinePlanner
from Pplan.Sampling.trajectory_tree import TrajTree
from cv2 import CCL_WU

import torch
import numpy as np
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.utils.tree import Tree
from typing import List
from enum import IntEnum

HOMOTOPY_THRESHOLD = np.pi/6
class HomotopyType(IntEnum):
    """
    Holds environment types - one per environment class.
    These act as identifiers for different environments.
    """
    STILL = 0
    CW = 1
    CCW = -1

HCList = [int(v) for v in HomotopyType]

    

class HomotopyTree(Tree):
    def __init__(self, homotopy, ego_xp, parent, depth,static_idx=None):
        self.homotopy = homotopy
        self.ego_xp = ego_xp
        self.opt_prob = None
        self.children = list()
        self.parent = parent
        self.static_idx = static_idx
        # self.branching_idx = None
        if parent is not None:
            parent.expand(self)
        self.depth = depth
        self.attribute = dict()
    def get_label(self):

        if self.static_idx is not None:
            return str(self.homotopy[~self.static_idx].cpu().numpy().astype(int))
        else:
            return str(self.homotopy.cpu().numpy().astype(int))

def mag_integral(path0,path1):
    if isinstance(path0,torch.Tensor):
        delta_path = path0-path1
        angle = torch.atan2(delta_path[...,1],delta_path[...,0])
        delta_angle = GeoUtils.round_2pi(angle[...,1:]-angle[...,:-1])
        angle_diff = torch.sum(delta_angle,dim=-1)
    elif isinstance(path0,torch.ndarray):
        delta_path = path0-path1
        angle = np.arctan2(delta_path[...,1],delta_path[...,0])
        delta_angle = GeoUtils.round_2pi(angle[...,1:]-angle[...,:-1])
        angle_diff = np.sum(delta_angle,axis=-1)
    return angle_diff

def identify_homotopy(ego_path:torch.Tensor, obj_paths:torch.Tensor,threshold = HOMOTOPY_THRESHOLD):
    """Identifying homotopy classes for the ego

    Args:
        ego_path (torch.Tensor): B x T x 2
        obj_paths (torch.Tensor): B x M x N x T x 2
    """
    b,M,N = obj_paths.shape[:3]
    angle_diff = mag_integral(ego_path[:,None,None],obj_paths)
    homotopy = torch.zeros([b,M,N],device=ego_path.device)
    homotopy[angle_diff>=threshold] = HomotopyType.CCW
    homotopy[angle_diff<=-threshold] = HomotopyType.CW
    homotopy[(angle_diff>-threshold) & (angle_diff<threshold)] = HomotopyType.STILL

    return angle_diff,homotopy

def grouping_homotopy_to_tree(homotopy, ego_xp, parent, depth=0, static_idx=None, strategy="greedy_balanced"):
    """Grouping list of homotopy tuples into a tree

    Args:
        homotopy (torch.Tensor): list of homotopy tuples [B x N]
        ego_xp (torch.Tensor): corresponding ego trajectory initializations [B x T x 4]
    """
    assert homotopy.shape[0]==ego_xp.shape[0]
    b, N = homotopy.shape
    if isinstance(homotopy,torch.Tensor):
        count = torch.zeros([N,len(HomotopyType)]).to(homotopy.device)
    else:
        count = np.zeros([N,len(HomotopyType)])
    for i,hc in enumerate(HomotopyType):
        count[:,i]=(homotopy==hc).sum(0)
    if static_idx is None:
        static_idx=((count==0).sum(1)==len(HomotopyType)-1)
    current_homotopy = (count>0)
    homotopy_tree = HomotopyTree(current_homotopy, ego_xp, parent, depth,static_idx)
    if b==1:
        return homotopy_tree
    if strategy=="greedy_balanced":
        if isinstance(homotopy,torch.Tensor):
            score = b*(len(HomotopyType)-1)-(torch.abs(count[:,0]-count[:,1])+torch.abs(count[:,0]-count[:,2])+torch.abs(count[:,1]-count[:,2]))
        else:
            score = b*(len(HomotopyType)-1)-(np.abs(count[:,0]-count[:,1])+np.abs(count[:,0]-count[:,2])+np.abs(count[:,1]-count[:,2]))
    else:
        raise NotImplementedError
    branch_idx = score.argmax()
    index = (-count[branch_idx]).argsort()

    for i in index:
        if count[branch_idx,i]>0:
            hc = HCList[i]
            selection = homotopy[:,branch_idx]==hc
            homotopy_subset = homotopy[selection]
            ego_xp_subset = ego_xp[selection]
            _ = grouping_homotopy_to_tree(homotopy_subset,ego_xp_subset,homotopy_tree,depth+1,static_idx,strategy=strategy)
    return homotopy_tree


            
    # for k in range()
    # if current_homotopy[idx]
    
        
    

def test():
    ego_traj = torch.cat((torch.zeros(10,1),torch.linspace(-5,5,10).unsqueeze(-1)),-1)
    obj1 = torch.cat((torch.ones(10,1),torch.zeros(10,1)),-1)
    obj2 = torch.cat((-torch.ones(10,1),torch.zeros(10,1)),-1)
    obj_traj = torch.stack((obj1,obj2),0)
    angle_diff,homotopy = identify_homotopy(ego_traj.unsqueeze(0),obj_traj.unsqueeze(0))
    print(angle_diff)

if __name__ =="__main__":
    test()
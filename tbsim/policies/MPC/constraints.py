import torch
import numpy as np
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.MPC.homotopy import HomotopyType,HOMOTOPY_THRESHOLD


def Rectangle_free_region_4(xyh: torch.Tensor, LW: torch.Tensor):
    """generate 4 disjoint free spaces around a rectangle

    Args:
        xyh (torch.Tensor): [B,3]
        LW (torch.Tensor): [B,2]

    Returns:
        A: [B x 4 x 3 x 2]
        b: [B x 4 x 3]
    """
    device = xyh.device
    x, y, h = xyh.unbind(-1)
    L, W = LW.unbind(-1)
    bs = x.shape[0]
    A0 = torch.tensor([[0., -1.0], [-1., -1.], [1., -1.]],device=device)
    A1 = torch.tensor([[1.0, 0.], [1., 1.], [1., -1.]],device=device)
    A2 = torch.tensor([[0., 1.0], [-1., 1.], [1., 1.]],device=device)
    A3 = torch.tensor([[-1.0, 0.], [-1., 1.], [-1., -1.]],device=device)

    A = (
        torch.stack((A0, A1, A2, A3), 0).unsqueeze(0).repeat_interleave(bs, 0)
    )  # B x 4 x 3 x 2

    b0 = torch.stack((-W / 2, L / 2 - W / 2, L / 2 - W / 2), -1)
    b1 = torch.stack((-L / 2, W / 2 - L / 2, W / 2 - L / 2), -1)
    b2 = torch.stack((-W / 2, L / 2 - W / 2, L / 2 - W / 2), -1)
    b3 = torch.stack((-L / 2, W / 2 - L / 2, W / 2 - L / 2), -1)

    b = torch.stack((b0, b1, b2, b3), 1)  # B x 4 x 3

    RotM = torch.cat(
        (
            torch.stack((torch.cos(h), torch.sin(h)), -1).unsqueeze(-2),
            torch.stack((-torch.sin(h), torch.cos(h)), -1).unsqueeze(-2),
        ),
        -2
    )  # b x 2 x 2
    RotM = RotM.unsqueeze(1).repeat_interleave(4, 1)  # B x 4 x 2 x 2
    offset = torch.stack((-x, -y), -1)[:, None].repeat_interleave(4, 1)

    A = A @ RotM
    b = b - (A @ offset.unsqueeze(-1)).squeeze(-1)
    return A, b

def left_shift(x):
    return torch.cat((x[...,1:],x[...,:1]),-1)

def right_shift(x):
    return torch.cat((x[...,-1:],x[...,:-1]),-1)

def Vehicle_coll_constraint(
    ego_xyh: torch.Tensor,
    ego_lw: torch.Tensor,
    obj_xyh: torch.Tensor,
    obj_lw: torch.Tensor,
    homotopy: torch.Tensor,
    active_flag: torch.Tensor = None,
    ignore_undecided=True,
    enforce_type="poly",
    angle_interval = 5,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_xyh (torch.Tensor): [B x T x 3]
        ego_lw (torch.Tensor): [B x 2]
        obj_xyh (torch.Tensor): [B x N x T x 3]
        obj_lw (torch.Tensor): [B x N x 2]
        active_flag (torch.Tensor): [B x N x T x 4] (4 comes from the number of regions for vehicle free space)
        homotopy (torch.Tensor[torch.bool]): [B x N x 3] (3 comes from the number of homotopy classes)
    """
    bs, Na, T = obj_xyh.shape[:3]
    xe,ye,he = ego_xyh.unbind(-1)
    ho = obj_xyh[...,2]
    # ego_xyh_tiled = ego_xyh.repeat_interleave(Na, 0)
    # ego_lw_tiled = ego_lw.repeat_interleave(Na, 0)
    obj_xyh_tiled = obj_xyh.reshape([-1, 3])
    obj_lw_tiled = obj_lw.repeat_interleave(T,1).reshape(-1,2)
    # A, b = TensorUtils.reshape_dimensions(Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled), 0, 1, (bs,Na,T))
    A, b = Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled)
    A = A.reshape([bs,Na,T,4,3,2])
    b = b.reshape([bs,Na,T,4,3])

    # number of free regions
    M = A.shape[-3]

    cornersX = torch.kron(ego_lw[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(ego_xyh.device))

    cornersY = torch.kron(ego_lw[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(ego_xyh.device))
    corners = torch.stack([cornersX, cornersY], dim=-1).reshape(bs,4,2)
    corners = GeoUtils.batch_rotate_2D(corners.unsqueeze(1), he.unsqueeze(-1).repeat_interleave(4, dim=-1))+ego_xyh[...,None,:2].repeat_interleave(4,-2) # bxTx4x2
    corner_constr = b.unsqueeze(-1)-A@corners[:,None,:,None].transpose(-1,-2) # b x Na x T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.max(3)[0] # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_xyh[:,None,:,None,:2,None]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    if enforce_type=="poly":
        if active_flag is None:
            current_region = center_constr.min(-1)[0].argmax(-1)
            current_flag = torch.zeros([bs,Na,T,M],device=ego_xyh.device,dtype=torch.bool)
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
        angle = GeoUtils.round_2pi(torch.atan2(delta_path[...,1],delta_path[...,0]))
        angle_constr = -10.0*torch.ones_like(angle)
        # for homotopy CCW, angle should be larger than 0
        angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,2].unsqueeze(-1),angle,-10))
        # for homotopy CW, angle should be less than 0
        angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,1].unsqueeze(-1),-angle,-10))
        # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
        angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,0].unsqueeze(-1),HOMOTOPY_THRESHOLD-angle.cumsum(-1).abs(),-10))

        center_constr = torch.cat((center_constr.reshape(bs,Na,-1),angle_constr),-1)

    
    
    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = torch.kron(obj_lw[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(ego_xyh.device))
    cornersY = torch.kron(obj_lw[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(ego_xyh.device))
    corners = torch.stack([cornersX, cornersY], dim=-1).reshape(bs,Na,4,2)
    RotM_obj = torch.cat(
        (
            torch.stack((torch.cos(ho), -torch.sin(ho)), -1).unsqueeze(-2),
            torch.stack((torch.sin(ho), torch.cos(ho)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x Na x T x 2 x 2
    RotM = torch.cat(
        (
            torch.stack((torch.cos(he), torch.sin(he)), -1).unsqueeze(-2),
            torch.stack((-torch.sin(he), torch.cos(he)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x T x 2 x 2
    corners = (RotM_obj.unsqueeze(3).repeat_interleave(4,3)@corners.unsqueeze(2).repeat_interleave(T,2).unsqueeze(-1)).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,None,:,None]@(corners+obj_xyh[...,None,:2]-ego_xyh[:,None,:,None,:2]).unsqueeze(-1)).squeeze(-1)
    obj_corner_constr = torch.maximum(torch.abs(corners[...,0])-ego_lw[:,None,None,None,0],torch.abs(corners[...,1])-ego_lw[:,None,None,None,1])
    if ignore_undecided:
        corner_constr[undecided] = 10.0
        center_constr[undecided] = 10.0
        obj_corner_constr[undecided] = 10.0
    constr = torch.cat((corner_constr.reshape(bs,-1),center_constr.reshape(bs,-1),obj_corner_constr.reshape(bs,-1)),-1)
    return constr


def Vehicle_coll_constraint_simple(
    ego_xyh: torch.Tensor,
    ego_lw: torch.Tensor,
    obj_xyh: torch.Tensor,
    obj_lw: torch.Tensor,
    homotopy: torch.Tensor,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_xyh (torch.Tensor): [B x T x 3]
        ego_lw (torch.Tensor): [B x 2]
        obj_xyh (torch.Tensor): [B x N x T x 3]
        obj_lw (torch.Tensor): [B x N x 2]
        active_flag (torch.Tensor): [B x N x T x 4] (4 comes from the number of regions for vehicle free space)
        homotopy (torch.Tensor[torch.bool]): [B x N x 3] (3 comes from the number of homotopy classes)
    """
    bs, Na, T = obj_xyh.shape[:3]
    xe,ye,he = ego_xyh.unbind(-1)
    ho = obj_xyh[...,2]
    obj_xyh_tiled = obj_xyh.reshape([-1, 3])
    obj_lw_tiled = obj_lw.repeat_interleave(T,1).reshape(-1,2)
    A, b = Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled)
    A = A.reshape([bs,Na,T,4,3,2])
    b = b.reshape([bs,Na,T,4,3])

    # number of free regions
    M = A.shape[-3]

    cornersX = torch.kron(ego_lw[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(ego_xyh.device))

    cornersY = torch.kron(ego_lw[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(ego_xyh.device))
    corners = torch.stack([cornersX, cornersY], dim=-1).reshape(bs,4,2)
    corners = GeoUtils.batch_rotate_2D(corners.unsqueeze(1), he.unsqueeze(-1).repeat_interleave(4, dim=-1))+ego_xyh[...,None,:2].repeat_interleave(4,-2) # bxTx4x2
    corner_constr = b.unsqueeze(-1)-A@corners[:,None,:,None].transpose(-1,-2) # b x Na x T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.max(3)[0] # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_xyh[:,None,:,None,:2,None]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    # if enforce_type=="poly":

    current_region = center_constr.min(-1)[0].argmax(-1)
    current_flag = torch.zeros([bs,Na,T,M],device=ego_xyh.device,dtype=torch.bool)
    current_flag.scatter_(-1,current_region.unsqueeze(-1),1)
    active_flag = current_flag.clone()

    # for homotopy number 1 (CW), left shift the current flag
    active_flag[...,1:,:] = active_flag[...,1:,:] | (left_shift(current_flag[...,:-1,:]) & homotopy[...,1:2].unsqueeze(-2))
    # for homotopy number 2 (CCW), right shift the current flag
    active_flag[...,1:,:] = active_flag[...,1:,:] | (right_shift(current_flag[...,:-1,:]) & homotopy[...,2:3].unsqueeze(-2))
        

    center_constr.masked_fill_(~active_flag.unsqueeze(-1),-10)  # mask out regions that are not active
    center_constr = center_constr.max(2)[0]


    
    
    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = torch.kron(obj_lw[..., 0] + offsetX, torch.tensor([0.5, 0.5, -0.5, -0.5]).to(ego_xyh.device))
    cornersY = torch.kron(obj_lw[..., 1] + offsetY, torch.tensor([0.5, -0.5, 0.5, -0.5]).to(ego_xyh.device))
    corners = torch.stack([cornersX, cornersY], dim=-1).reshape(bs,Na,4,2)
    RotM_obj = torch.cat(
        (
            torch.stack((torch.cos(ho), -torch.sin(ho)), -1).unsqueeze(-2),
            torch.stack((torch.sin(ho), torch.cos(ho)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x Na x T x 2 x 2
    RotM = torch.cat(
        (
            torch.stack((torch.cos(he), torch.sin(he)), -1).unsqueeze(-2),
            torch.stack((-torch.sin(he), torch.cos(he)), -1).unsqueeze(-2),
        ),
        -2,
    )  # b x T x 2 x 2
    corners = (RotM_obj.unsqueeze(3).repeat_interleave(4,3)@corners.unsqueeze(2).repeat_interleave(T,2).unsqueeze(-1)).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[:,None,:,None]@(corners+obj_xyh[...,None,:2]-ego_xyh[:,None,:,None,:2]).unsqueeze(-1)).squeeze(-1)
    obj_corner_constr = torch.maximum(torch.abs(corners[...,0])-ego_lw[:,None,None,None,0],torch.abs(corners[...,1])-ego_lw[:,None,None,None,1])
    corner_constr[undecided] = 10.0
    center_constr[undecided] = 10.0
    obj_corner_constr[undecided] = 10.0
    constr = torch.cat((corner_constr.reshape(bs,-1),center_constr.reshape(bs,-1),obj_corner_constr.reshape(bs,-1)),-1)
    return constr

def pedestrian_coll_constraint(
    ego_xyh: torch.Tensor,
    ego_lw: torch.Tensor,
    obj_xyh: torch.Tensor,
    obj_R: torch.Tensor,
    homotopy: torch.Tensor,
    angle_interval = 5,
    ignore_undecided=True,
    offsetR=0.0,
):
    """generate collision avoidance constraint for pedestrain objects

    Args:
        ego_xyh (torch.Tensor): [B x T x 3]
        ego_lw (torch.Tensor): [B x 2]
        obj_xyh (torch.Tensor): [B x N x T x 3]
        obj_R (torch.Tensor): [B x N x 1]
    """
    bs,Na,T = obj_xyh.shape[:3]
    theta = ego_xyh[..., 2]
    dx = GeoUtils.batch_rotate_2D(obj_xyh[..., 0:2] - ego_xyh[:,None,:, 0:2], -theta.unsqueeze(1))
    
    marginxy = torch.stack((torch.abs(dx[...,0])-ego_lw[:,None,None,0],torch.abs(dx[...,1])-ego_lw[:,None,None,1]),-1)
    marginp = marginxy.clip(min=0)
    hypot = torch.norm(marginp,dim=-1)
    hypot.masked_fill_((marginxy<0).any(-1),-1e4)
    margin = torch.maximum(marginxy.max(-1)[0],hypot)
    undecided = homotopy.sum(-1)>1

    coll_constr = (margin-obj_R-offsetR)
    delta_path = ego_xyh[:,None,::angle_interval,:2]-obj_xyh[:,:,::angle_interval,:2]
    angle = GeoUtils.round_2pi(torch.arctan2(delta_path[...,1],delta_path[...,0]))
    angle_constr = -10.0*torch.ones_like(angle)
    # for homotopy CCW, angle should be larger than 0
    angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,2].unsqueeze(-1),angle,-10))
    # for homotopy CW, angle should be less than 0
    angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,1].unsqueeze(-1),-angle,-10))
    # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
    angle_constr = torch.maximum(angle_constr,torch.where(homotopy[...,0].unsqueeze(-1),HOMOTOPY_THRESHOLD-angle.cumsum(-1).abs(),-10))

    if ignore_undecided:
        coll_constr[undecided] = 10.0
        angle_constr[undecided] = 10.0

    
    return torch.cat((coll_constr.reshape(bs,-1),angle_constr.reshape(bs,-1)),-1)

def polyline_constr(
    ego_xyh: torch.Tensor,
    ego_lw: torch.Tensor,
    polyline: torch.Tensor,
    direction: torch.Tensor,
    margin=0.0,
):
    """generate boundary constraint given polyline boundaries

    Args:
        ego_xyh (torch.Tensor): [B x T x 3]
        ego_lw (torch.Tensor): [B x 2]
        polyline (torch.Tensor): [B x N x L x 3]
        direction (torch.Tensor): [B x N]: 1 to stay on the right of line, -1 to stay on the left of the line
        margin (float, default to 0.0): allowed margin
    """
    bs,T = ego_xyh.shape[:2]
    N = polyline.shape[1]
    he = ego_xyh[...,2]
    L,W = ego_lw.unbind(-1)
    margin = L/2*torch.sin(he).abs()+W/2*torch.cos(he).abs()
    delta_x,delta_y,_ = GeoUtils.batch_proj(TensorUtils.join_dimensions(ego_xyh,0,2).repeat_interleave(N,0), TensorUtils.join_dimensions(polyline.repeat_interleave(T,0),0,2))
    idx = delta_x.abs().argmin(1)
    
    delta_y = torch.gather(delta_y,1,idx.unsqueeze(-1)).squeeze(-1).reshape(bs,T,N)
    return delta_y*direction.unsqueeze(1)-margin.unsqueeze(-1)


def test_region():
    import matplotlib.pyplot as plt
    import polytope as pc
    from matplotlib.patches import Polygon

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 1.0, 1.0])
    theta = torch.tensor([0, 0.1, -0.5])
    L = torch.tensor([3.0, 3.1, 4])
    W = torch.tensor([2, 2.1, 1.9])
    xyh = torch.stack((x, y, theta), -1)
    LW = torch.stack((L, W), -1)
    A, b = Rectangle_free_region_4(xyh, LW)
    A = A.numpy()
    b = b.numpy()
    A0 = np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]])
    b0 = np.array([10.0, 10.0, 10.0, 10.0])
    p0 = pc.Polytope(A0, b0)
    # b1 = np.array([15.,15.,-5.,-5.])
    fig, ax = plt.subplots()
    k = 2
    for i in range(4):
        A1 = np.vstack((A[k, i], A0))
        b1 = np.append(b[k, i], b0)
        p1 = pc.Polytope(A1, b1)
        V = pc.extreme(p1)
        patch = Polygon(V)
        ax.add_patch(patch)

    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])
    plt.show()
    print("done")


def test_gradient():
    from torch.autograd.functional import jacobian

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 1.0, 1.0])
    theta = torch.tensor([0, 0.1, -0.5])
    L = torch.tensor([3.0, 3.1, 4])
    W = torch.tensor([2, 2.1, 1.9])
    xyh = torch.stack((x, y, theta), -1)
    LW = torch.stack((L, W), -1)
    A, b = Rectangle_free_region_4(xyh, LW)
    x0 = torch.tensor([4.0, 5.0])
    constraint = b[0] - A[0] @ x0

    def test_fun(xyh, LW, x0):
        A, b = Rectangle_free_region_4(xyh, LW)
        constraint = b[0] - A[0] @ x0
        return constraint[2, 1]

    
    jacobian(test_fun, (xyh, LW, x0))


def test_constraint():
    from torch.autograd.functional import jacobian

    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 1.0, 1.0])
    theta = torch.tensor([-np.pi/3, 0.1, -0.5])
    L = torch.tensor([3.0, 3.1, 4])
    W = torch.tensor([2, 2.1, 1.9])
    obj_xyh = torch.stack((x, y, theta), -1)
    obj_xyh[0,0]=4+np.sqrt(3)+0.75+np.sqrt(3)/2
    obj_xyh[0,1]=5+1-0.75*np.sqrt(3)+0.5
    obj_LW = torch.stack((L, W), -1)
    x0 = torch.tensor([4.0, 5.0, np.pi/6])
    ego_LW = torch.tensor([4.0, 2.5])
    active_flag = torch.tensor(
        [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool
    )
    T = 6
    bs = 5
    
    homotopy = torch.tensor([[1,0,0],[1,1,0],[0,0,1]],dtype=torch.bool)

    scripted_fn = torch.jit.script(Vehicle_coll_constraint_simple, example_inputs=[(x0[None,None,:].repeat_interleave(T,1).repeat_interleave(bs,0),ego_LW[None,:].repeat_interleave(bs,0),obj_xyh[None,:,None].repeat_interleave(T,2).repeat_interleave(bs,0),obj_LW[None,:].repeat_interleave(bs,0),homotopy[None,...].repeat_interleave(bs,0),0.0,0.0)])
    # func = lambda ego_xyh, obj_xyh: Vehicle_coll_constraint(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_LW[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0),enforce_type="angle")
    func = lambda ego_xyh, obj_xyh: scripted_fn(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_LW[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0))
    inputs = (x0[None,None,:].repeat_interleave(T,1).repeat_interleave(bs,0),obj_xyh[None,:,None].repeat_interleave(T,2).repeat_interleave(bs,0))
    constr = func(*inputs)
    Jacobian = jacobian(func, inputs,vectorize=True)
    obj_R = obj_LW[...,[1]]
    func = lambda ego_xyh, obj_xyh: pedestrian_coll_constraint(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_R[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0))
    constrp = func(*inputs)
    Jacobianp = jacobian(func, inputs)
    line1 = torch.stack([torch.linspace(-10.,30,20),torch.randn(20),torch.zeros(20)],-1)
    line2 = torch.stack([torch.linspace(-10.,30,20),torch.randn(20)+8,torch.zeros(20)],-1)
    polyline = torch.stack((line1,line2),0)
    direction = torch.tensor([[1,-1]]).repeat_interleave(2,0)
    constr = polyline_constr(x0[None,None,:].repeat_interleave(2,0).repeat_interleave(5,1),ego_LW.unsqueeze(0),polyline[None,:].repeat_interleave(2,0),direction)
    print("hello")

if __name__ == "__main__":
    test_constraint()

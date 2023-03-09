import numpy as np
import jax.numpy as jnp
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.policies.MPC.homotopy import HomotopyType,HOMOTOPY_THRESHOLD
from jax import jacfwd, jacrev


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


def Vehicle_coll_constraint_simple(
    ego_xyh: jnp.array,
    ego_lw: jnp.array,
    obj_xyh: jnp.array,
    obj_lw: jnp.array,
    homotopy: jnp.array,
    offsetX=0.0,
    offsetY=0.0,
):
    """generate collision avoidance constraint for vehicle objects

    Args:
        ego_xyh (jnp.array): [T x 3]
        ego_lw (jnp.array): [2]
        obj_xyh (jnp.array): [N x T x 3]
        obj_lw (jnp.array): [N x 2]
        homotopy (jnp.array[bool]): [N x 3] (3 comes from the number of homotopy classes)
    """
    Na, T = obj_xyh.shape[:2]
    he = ego_xyh[...,2]
    ho = obj_xyh[...,2]
    obj_xyh_tiled = obj_xyh.reshape([-1, 3])
    obj_lw_tiled = obj_lw.repeat(T,0).reshape(-1,2)
    
    M = 4
    A, b = Rectangle_free_region_4(obj_xyh_tiled, obj_lw_tiled)
    A = A.reshape([Na,T,M,3,2])
    b = b.reshape([Na,T,M,3])

    # number of free regions
    

    # cornersX = jnp.kron(ego_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))

    # cornersY = jnp.kron(ego_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    cornersX = (ego_lw[..., 0] + offsetX)*jnp.array([0.5, 0.5, -0.5, -0.5])
    cornersY = (ego_lw[..., 1] + offsetY)*jnp.array([0.5, -0.5, 0.5, -0.5])
    corners = jnp.stack([cornersX, cornersY], axis=-1)
    corners = GeoUtils.batch_rotate_2D(corners[np.newaxis,:], he[:,np.newaxis].repeat(4, axis=1)).reshape(T,4,2)+ego_xyh[...,np.newaxis,:2].repeat(4,-2) # Tx4x2
    corner_constr = b[...,np.newaxis]-A@(corners.transpose([0,2,1])[np.newaxis,:,np.newaxis]) # Na x T x M(region) x 3(hyperplane for each region) x 4(corners)
    corner_constr = corner_constr.min(3).max(2) # corners only need to stay in one of the 4 regions
    center_constr = b-(A@ego_xyh[np.newaxis,:,np.newaxis,:2,np.newaxis]).squeeze(-1)
    # ignore agents with more than 1 homotopies
    undecided = homotopy.sum(-1)>1
    
    min_constr = center_constr.min(-1)

    current_flag = min_constr>=min_constr.max(-1)[...,np.newaxis]
    active_flag = current_flag.clone()

    # for homotopy number 1 (CW), left shift the current flag
    active_flag = active_flag.at[...,1:,:].max(left_shift(current_flag[...,:-1,:]) & homotopy[:,np.newaxis,1:2])
    # for homotopy number 2 (CCW), right shift the current flag
    active_flag = active_flag.at[...,1:,:].max(right_shift(current_flag[...,:-1,:]) & homotopy[:,np.newaxis,2:3])

    mask = jnp.ones_like(center_constr)*1e4*active_flag[...,np.newaxis]+jnp.ones_like(center_constr)*-10*~active_flag[...,np.newaxis]

    center_constr = jnp.minimum(center_constr,mask)  # mask out regions that are not active
    center_constr = center_constr.min(3).max(2)


    
    
    # now calculate constraint that the objects' corners do not collide with the ego    
    cornersX = jnp.kron(obj_lw[..., 0] + offsetX, jnp.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = jnp.kron(obj_lw[..., 1] + offsetY, jnp.array([0.5, -0.5, 0.5, -0.5]))
    corners = jnp.stack([cornersX, cornersY], axis=-1).reshape(Na,4,2)
    RotM_obj = jnp.stack(
        (
            jnp.stack((jnp.cos(ho), -jnp.sin(ho)), -1),
            jnp.stack((jnp.sin(ho), jnp.cos(ho)), -1),
        ),
        -2,
    )  # Na x T x 2 x 2
    RotM = jnp.stack(
        (
            jnp.stack((jnp.cos(he), jnp.sin(he)), -1),
            jnp.stack((-jnp.sin(he), jnp.cos(he)), -1),
        ),
        -2,
    )  # T x 2 x 2
    corners = (RotM_obj[:,:,np.newaxis]@corners[:,np.newaxis,:,:,np.newaxis]).squeeze(-1)
    # transform corners into ego coordinate
    corners = (RotM[np.newaxis,:,np.newaxis]@(corners+obj_xyh[:,:,np.newaxis,:2]-ego_xyh[np.newaxis,:,np.newaxis,:2])[...,np.newaxis]).squeeze(-1)
    obj_corner_constr = jnp.maximum(jnp.abs(corners[...,0])-ego_lw[0],jnp.abs(corners[...,1])-ego_lw[1])
    corner_constr = corner_constr.at[undecided].set(10.0)
    center_constr = center_constr.at[undecided].set(10.0)
    obj_corner_constr = obj_corner_constr.at[undecided].set(10.0)
    constr = jnp.concatenate((corner_constr.flatten(),center_constr.flatten(),obj_corner_constr.flatten()),-1)
    return constr

def pedestrian_coll_constraint(
    ego_xyh: jnp.array,
    ego_lw: jnp.array,
    obj_xyh: jnp.array,
    obj_R: jnp.array,
    homotopy: jnp.array,
    angle_interval = 5,
    ignore_undecided=True,
    offsetR=0.0,
):
    """generate collision avoidance constraint for pedestrain objects

    Args:
        ego_xyh (jnp.array): [B x T x 3]
        ego_lw (jnp.array): [B x 2]
        obj_xyh (jnp.array): [B x N x T x 3]
        obj_R (jnp.array): [B x N x 1]
    """
    bs,Na,T = obj_xyh.shape[:3]
    theta = ego_xyh[..., 2]
    dx = GeoUtils.batch_rotate_2D(obj_xyh[..., 0:2] - ego_xyh[:,None,:, 0:2], -theta.unsqueeze(1))
    
    marginxy = jnp.stack((jnp.abs(dx[...,0])-ego_lw[:,None,None,0],jnp.abs(dx[...,1])-ego_lw[:,None,None,1]),-1)
    marginp = marginxy.clip(min=0)
    hypot = jnp.linalg.norm(marginp,axis=-1)
    hypot.masked_fill_((marginxy<0).any(-1),-1e4)
    margin = jnp.maximum(marginxy.max(-1)[0],hypot)
    undecided = homotopy.sum(-1)>1

    coll_constr = (margin-obj_R-offsetR)
    delta_path = ego_xyh[:,None,::angle_interval,:2]-obj_xyh[:,:,::angle_interval,:2]
    angle = GeoUtils.round_2pi(jnp.arctan2(delta_path[...,1],delta_path[...,0]))
    angle_constr = -10.0*jnp.ones_like(angle)
    # for homotopy CCW, angle should be larger than 0
    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,2].unsqueeze(-1),angle,-10))
    # for homotopy CW, angle should be less than 0
    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,1].unsqueeze(-1),-angle,-10))
    # for homotopy STILL, angle absolute value should be less than HOMOTOPY_THRESHOLD
    angle_constr = jnp.maximum(angle_constr,jnp.where(homotopy[...,0].unsqueeze(-1),HOMOTOPY_THRESHOLD-angle.cumsum(-1).abs(),-10))

    if ignore_undecided:
        coll_constr[undecided] = 10.0
        angle_constr[undecided] = 10.0

    
    return jnp.concatenate((coll_constr.reshape(bs,-1),angle_constr.reshape(bs,-1)),-1)

def polyline_constr(
    ego_xyh: jnp.array,
    ego_lw: jnp.array,
    polyline: jnp.array,
    direction: jnp.array,
    margin=0.0,
):
    """generate boundary constraint given polyline boundaries

    Args:
        ego_xyh (jnp.array): [B x T x 3]
        ego_lw (jnp.array): [B x 2]
        polyline (jnp.array): [B x N x L x 3]
        direction (jnp.array): [B x N]: 1 to stay on the right of line, -1 to stay on the left of the line
        margin (float, default to 0.0): allowed margin
    """
    bs,T = ego_xyh.shape[:2]
    N = polyline.shape[1]
    he = ego_xyh[...,2]
    L,W = ego_lw.unbind(-1)
    margin = L/2*jnp.sin(he).abs()+W/2*jnp.cos(he).abs()
    delta_x,delta_y,_ = GeoUtils.batch_proj(TensorUtils.join_axisensions(ego_xyh,0,2).repeat_interleave(N,0), TensorUtils.join_axisensions(polyline.repeat_interleave(T,0),0,2))
    idx = delta_x.abs().argmin(1)
    
    delta_y = jnp.take_along_axis(delta_y,idx.unsqueeze(-1),1).squeeze(-1).reshape(bs,T,N)
    return delta_y*direction.unsqueeze(1)-margin.unsqueeze(-1)


def test_region():
    import matplotlib.pyplot as plt
    import polytope as pc
    from matplotlib.patches import Polygon

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.0, 1.0, 1.0])
    theta = jnp.array([0, 0.1, -0.5])
    L = jnp.array([3.0, 3.1, 4])
    W = jnp.array([2, 2.1, 1.9])
    xyh = jnp.stack((x, y, theta), -1)
    LW = jnp.stack((L, W), -1)
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



def test_constraint():

    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([1.0, 1.0, 1.0])
    theta = jnp.array([-np.pi/3, 0.1, -0.5])
    L = jnp.array([3.0, 3.1, 4])
    W = jnp.array([2, 2.1, 1.9])
    obj_xyh = jnp.stack((x, y, theta), -1)
    obj_xyh = obj_xyh.at[0,0].set(4+np.sqrt(3)+0.75+np.sqrt(3)/2)
    obj_xyh = obj_xyh.at[0,1].set(5+1-0.75*np.sqrt(3)+0.5)
    obj_LW = jnp.stack((L, W), -1)
    x0 = jnp.array([4.0, 5.0, np.pi/6])
    
    ego_LW = jnp.array([4.0, 2.5])
        
    T = 6
    bs = 5
    ego_xyh = jnp.expand_dims(x0,0).repeat(T,0)
    obj_xyh = jnp.expand_dims(obj_xyh,1).repeat(T,1)
    homotopy = jnp.array([[1,0,0],[1,1,0],[0,0,1]],dtype=bool)
    constr = Vehicle_coll_constraint_simple(ego_xyh,ego_LW,obj_xyh,obj_LW,homotopy)
    func = lambda x: Vehicle_coll_constraint_simple(x[0],ego_LW,x[1],obj_LW,homotopy)
    J = jacfwd(func)((ego_xyh,obj_xyh))
    # # func = lambda ego_xyh, obj_xyh: Vehicle_coll_constraint(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_LW[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0),enforce_type="angle")
    # func = lambda ego_xyh, obj_xyh: scripted_fn(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_LW[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0))
    # inputs = (x0[None,None,:].repeat_interleave(T,1).repeat_interleave(bs,0),obj_xyh[None,:,None].repeat_interleave(T,2).repeat_interleave(bs,0))
    # constr = func(*inputs)
    # Jacobian = jacobian(func, inputs,vectorize=True)
    # obj_R = obj_LW[...,[1]]
    # func = lambda ego_xyh, obj_xyh: pedestrian_coll_constraint(ego_xyh,ego_LW[None,:].repeat_interleave(bs,0),obj_xyh,obj_R[None,:].repeat_interleave(bs,0),homotopy=homotopy[None,...].repeat_interleave(bs,0))
    # constrp = func(*inputs)
    # Jacobianp = jacobian(func, inputs)
    # line1 = jnp.stack([torch.linspace(-10.,30,20),torch.randn(20),jnp.zeros(20)],-1)
    # line2 = jnp.stack([torch.linspace(-10.,30,20),torch.randn(20)+8,jnp.zeros(20)],-1)
    # polyline = jnp.stack((line1,line2),0)
    # direction = jnp.array([[1,-1]]).repeat_interleave(2,0)
    # constr = polyline_constr(x0[None,None,:].repeat_interleave(2,0).repeat_interleave(5,1),ego_LW.unsqueeze(0),polyline[None,:].repeat_interleave(2,0),direction)
    print("hello")

if __name__ == "__main__":
    test_constraint()

import numpy as np
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils


def get_edge(lane,dir,W=2.0,num_pts = None):

    if dir == "L":
        
        if lane.left_edge is not None:
            lane.left_edge = lane.left_edge.interpolate(num_pts)
            xy = lane.left_edge.xy
            if lane.left_edge.has_heading:
                h = lane.left_edge.h
            else:
                dxy = xy[1:]-xy[:-1]
                h = GeoUtils.round_2pi(np.arctan2(dxy[:,1],dxy[:,0]))
                h = np.hstack((h,h[-1]))
        else:
            lane.center = lane.center.interpolate(num_pts)
            angle = lane.center.h+np.pi/2
            offset = np.stack([W*np.cos(angle),W*np.sin(angle)],-1)
            xy = lane.center.xy+offset
            h = lane.center.h
    elif dir == "R":
        
        if lane.right_edge is not None:
            lane.right_edge = lane.right_edge.interpolate(num_pts)
            xy = lane.right_edge.xy
            if lane.right_edge.has_heading:
                h = lane.right_edge.h
            else:
                dxy = xy[1:]-xy[:-1]
                h = GeoUtils.round_2pi(np.arctan2(dxy[:,1],dxy[:,0]))
                h = np.hstack((h,h[-1]))
        else:
            lane.center = lane.center.interpolate(num_pts)
            angle = lane.center.h-np.pi/2
            offset = np.stack([W*np.cos(angle),W*np.sin(angle)],-1)
            xy = lane.center.xy+offset
            h = lane.center.h
    elif dir =="C":
        lane.center = lane.center.interpolate(num_pts)
        xy = lane.center.xy
        if lane.center.has_heading:
            h = lane.center.h
        else:
            dxy = xy[1:]-xy[:-1]
            h = GeoUtils.round_2pi(np.arctan2(dxy[:,1],dxy[:,0]))
            h = np.hstack((h,h[-1]))
    return xy,h

def get_bdry_xyh(lane1,lane2=None,dir="L",W=3.6,num_pts = 25):
    if lane2 is None:
        xy,h = get_edge(lane1,dir,W,num_pts*2)
    else:
        xy1,h1 = get_edge(lane1,dir,W,num_pts)
        xy2,h2 = get_edge(lane2,dir,W,num_pts)
        xy = np.concatenate((xy1,xy2),0)
        h = np.concatenate((h1,h2),0)
    return xy,h

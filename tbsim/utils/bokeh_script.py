import argparse
import numpy as np
import tbsim.utils.geometry_utils as GeoUtils
import tbsim.utils.tensor_utils as TensorUtils
import tbsim.utils.lane_utils as LaneUtils
from tbsim.policies.MPC.homotopy import HomotopyType,HOMOTOPY_THRESHOLD

XYH_INDEX = np.array([0,1,3])
import h5py
import pickle
import json
import pandas as pd  
from bokeh.io import curdoc  
from bokeh.layouts import row,column  
from bokeh.models import ColumnDataSource,LabelSet,PointDrawTool  
from bokeh.models.widgets import Slider, Paragraph,Button,CheckboxButtonGroup  
from bokeh.plotting import figure, show, output_file, output_notebook
from scipy.spatial import ConvexHull
from trajdata import MapAPI, VectorMap
from pathlib import Path

from bokeh.models import Range1d
import bokeh
import sys
from collections import defaultdict
import os
args = defaultdict(list)
argname = None
for x in sys.argv:
    if x.startswith("-"):
        argname=x[1:]
    else:
        if argname is not None:
            args[argname].append(x)

for k,v in args.items():
    if len(v)==1:
        args[k]=v[0]


assert "result_dir" in args
result_dir = args["result_dir"]

if "cache_path" in args:
    cache_path = Path(args["cache_path"]).expanduser()
else:
    cache_path = Path("~/.unified_data_cache").expanduser()
mapAPI = MapAPI(cache_path)

sim_info_path = os.path.join(result_dir,"sim_info.json")
sim_info = json.load(open(sim_info_path, "r"))

if "scene_name" in args:
    scene_name = args["scene_name"]
else:
    scene_name = sim_info["scene_index"][0]

if "episode" in args:
    ei = args["episode"]
else:
    ei = 0
sim_name = f"{scene_name}_{ei}"


map_name = sim_info["map_info"][scene_name]

vec_map = mapAPI.get_map(map_name, scene_cache=None)




hdf5_path = os.path.join(result_dir,"data.hdf5")
h5f = h5py.File(hdf5_path, "r")

trace_path = hdf5_path = os.path.join(result_dir,"trace.pkl")

with open(trace_path, "rb") as f:
    trace = pickle.load(f)


trace_offset = 11
def plot_lane(lane,plot,color="grey"):
    bdry_l,_ = LaneUtils.get_edge(lane,dir="L",num_pts=15)
    bdry_r,_ = LaneUtils.get_edge(lane,dir="R",num_pts=15)
    lane_center,_ = LaneUtils.get_edge(lane,dir="C",num_pts=15)
    bdry_xy = np.concatenate([bdry_l,np.flip(bdry_r,0)],0)
    patch_glyph = plot.patch(x=bdry_xy[:,0],y=bdry_xy[:,1],fill_alpha=0.5,color = color)
    centerline_glyph = plot.line(x=lane_center[:,0],y=lane_center[:,1],line_dash="dashed",line_width=2)
    return patch_glyph,centerline_glyph


def get_agent_edge(xy,h,extent):
    
    edges = np.array([[0.5,0.5],[0.5,-0.5],[-0.5,-0.5],[-0.5,0.5]])*extent[np.newaxis,:2]
    rotM = np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])
    edges = (rotM@edges[...,np.newaxis]).squeeze(-1)+xy[np.newaxis,:]
    return edges

def button_callback():
    sys.exit()  # Stop the server
plot = figure(name='base',height=1000, width=1000, title="traffic Animation",  
                tools="reset,save",toolbar_location="below",match_aspect=True)
plot.xgrid.grid_line_color = None
plot.ygrid.grid_line_color = None  
plot.axis.visible=False

sim_record = h5f[sim_name]
sim_trace = trace[sim_name]
plan_ts = np.array(list(trace[sim_name].keys()))
lanes = set()
lanecenter_glyph = dict()
lanepatch_glyph = dict()
agents = set()
plan_glyph = dict()

Na,T = sim_record["centroid"].shape[:2]

agent_id=["ego"]+[f"A{i}" for i in range(1,Na)]
palette = bokeh.palettes.Category20[20]
agent_color = ["blueviolet"] + [palette[i%20] for i in range(Na-1)]
agent_ds = dict()
agent_patch = dict()
agents = set()
agent_plan_ds=dict()
agent_plan_glyph = dict()

# setup patches

numModes = 5

for t in range(T):
    # plotting lanes
    xyz = np.hstack([sim_record["centroid"][0,t],np.zeros(1)])
    lanes_t = vec_map.get_lanes_within(xyz,100)
    for lane in lanes_t:
        if lane not in lanes:
            patch_glyph,centerline_glyph = plot_lane(lane,plot)
            lanecenter_glyph[lane] = centerline_glyph
            lanepatch_glyph[lane] = patch_glyph
            lanes.add(lane)

    track_ids = np.where((sim_record["centroid"][:,t]!=0).any(-1))[0]
    for id in track_ids:
        if id not in agents:
            agents.add(id)
            edges = get_agent_edge(sim_record["centroid"][id,t],sim_record["yaw"][id,t],sim_record["extent"][id,t])
            source = ColumnDataSource(data=dict(x=edges[:,0],y=edges[:,1]))
            agent_patch[id] = plot.patch(x="x",y="y",source=source,color=agent_color[id])
            agent_ds[id] = source
            # plotting plans
            
            plan_source = [ColumnDataSource(data=dict(x=np.random.randn(30),y=np.random.randn(30))) for i in range(numModes)]
            agent_plan_glyph[id] = [plot.line(x="x",y="y",source=plan_source[i],color=agent_color[id],line_width=2) for i in range(numModes)]
            agent_plan_ds[id] = plan_source
            
    xref_source = ColumnDataSource(data=dict(x=np.random.randn(30),y=np.random.randn(30)))
    xref_glyph = plot.line(x="x",y="y",source=xref_source,color=agent_color[0],line_width=1.5,line_dash="dashed")


slider = Slider(title="Sim time", value=0, start=0, end=T-1, step=1)

def update_data(attrname, old, new):  
    t = slider.value #holds the current time value of slider after updating the slider
    
    # update agent patches
    track_ids = np.where((sim_record["centroid"][:,t]!=0).any(-1))[0]
    last_update_idx = np.where(t+trace_offset>=plan_ts)[0].argmax().item()
    last_update_t = plan_ts[last_update_idx].item()
    world_from_agent = sim_record["world_from_agent"][0,last_update_t-trace_offset]
    if sim_trace[last_update_t]["obj_x"] is not None:
        obj_plan = GeoUtils.batch_nd_transform_points_np(sim_trace[last_update_t]["obj_x"][:,:,:2],world_from_agent[np.newaxis,:])
    else:
        obj_plan = None
    ego_plan = GeoUtils.batch_nd_transform_points_np(sim_trace[last_update_t]["ego_x"][:,:2],world_from_agent)
    if "xref" in sim_trace[last_update_t] and sim_trace[last_update_t]["xref"] is not None:
        xref = sim_trace[last_update_t]["xref"][...,:2]
        
        xref = GeoUtils.batch_nd_transform_points_np(xref,world_from_agent)
    else:
        xref = None
    for id in track_ids:
        edges = get_agent_edge(sim_record["centroid"][id,t],sim_record["yaw"][id,t],sim_record["extent"][id,t])
        agent_ds[id].data.update(dict(x=edges[:,0],y=edges[:,1]))
        if id==0:
            agent_plan_ds[id][0].data.update(dict(x=ego_plan[:,0],y=ego_plan[:,1]))
            if sim_trace[last_update_t]["ego_x"] is not None:
                if sim_trace[last_update_t]["ego_candidate_x"] is not None:
                    ego_candidate_trajs = GeoUtils.batch_nd_transform_points_np(sim_trace[last_update_t]["ego_candidate_x"][...,:2],world_from_agent[None,None,:])
                else:
                    ego_candidate_trajs = None
                for i in range(numModes-1):
                    if ego_candidate_trajs is not None and i<ego_candidate_trajs.shape[0]:
                        agent_plan_ds[id][i+1].data.update(dict(x=ego_candidate_trajs[i,:,0],y=ego_candidate_trajs[i,:,1]))
                    else:
                        agent_plan_ds[id][i+1].data.update(dict(x=[],y=[]))
            if xref is not None:
                xref_source.data.update(dict(x=xref[:,0],y=xref[:,1]))
        elif id in sim_trace[last_update_t]["track_ids"]:
            idx = np.where(sim_trace[last_update_t]["track_ids"]==id)[0].item()
            # print(obj_plan[idx,0,:2]-sim_record["centroid"][id,t])
            # print(obj_plan[idx,:,:2]-ego_plan)
            if obj_plan is not None:
                if idx<obj_plan.shape[0]:
                    agent_plan_ds[id][0].data.update(dict(x=obj_plan[idx,:,0],y=obj_plan[idx,:,1]))
                else:
                    agent_plan_ds[id][0].data.update(dict(x=[],y=[]))
        

    ego_xy = sim_record["centroid"][0,t]
    # plot.x_range=Range1d(ego_xy[0]-20,ego_xy[0]+20)
    plot.x_range.start = ego_xy[0]-50
    plot.x_range.end = ego_xy[0]+50
    plot.y_range.start = ego_xy[1]-50
    plot.y_range.end = ego_xy[1]+50



plot.x_range=Range1d(sim_record["centroid"][0,0,0]-50,sim_record["centroid"][0,0,0]+50)
plot.y_range=Range1d(sim_record["centroid"][0,0,1]-50,sim_record["centroid"][0,0,1]+50)

exit_button = Button(label="Stop", button_type="success",width=60)
exit_button.on_click(button_callback)
slider.on_change('value', update_data)


play_button = Button(label='► Play', width=60)
def animate_update():
    t = slider.value + 1 #gets value of slider + 1
    if t > slider.end:  
        t = 0  #if slider value+1 is above max, reset to 0
    slider.value = t  
#Update the label on the button once the button is clicked
global play_cb
def animate():  
    global play_cb
    if play_button.label == '► Play':  
        play_button.label = '❚❚ Pause'
        
        play_cb = curdoc().add_periodic_callback(animate_update, 100)  #50 is speed of animation
        # curdoc().remove_periodic_callback(animate_update)
    else:  
        play_button.label = '► Play'  
        curdoc().remove_periodic_callback(play_cb)
    
#callback when button is clicked.
play_button.on_click(animate)


layout = column(row(slider,exit_button,play_button),plot)  #add plot to layout

curdoc().add_root(layout)


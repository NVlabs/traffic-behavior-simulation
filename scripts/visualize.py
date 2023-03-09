import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
import pathlib
import json
from scipy.signal import savgol_filter
import os
import imageio
import glob

from l5kit.data import LocalDataManager
from l5kit.geometry import transform_points
from trajdata.simulation.sim_stats import calc_stats
from trajdata.simulation.sim_df_cache import SimulationDataFrameCache
from trajdata import AgentType, UnifiedDataset

from tbsim.utils.geometry_utils import get_box_world_coords_np
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_trajdata_cfg
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from tbsim.utils.vis_utils import COLORS, draw_agent_boxes
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.configs.eval_config import EvaluationConfig
import tbsim.utils.tensor_utils as TensorUtils
from PIL import Image, ImageDraw

import matplotlib.collections as mcoll
import matplotlib.patches as patches
import matplotlib.path as mpath


def colorline(
        ax, x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0,zorder=1):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,zorder=zorder)

    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def get_l5_rasterizer(dataset_path):
    exp_cfg = get_registered_experiment_config("l5_bc")
    exp_cfg.dataset_path = dataset_path
    os.environ["L5KIT_DATA_FOLDER"] = dataset_path

    l5_config = translate_l5kit_cfg(exp_cfg)
    l5_config["raster_params"]["raster_size"] = (500, 500)
    l5_config["raster_params"]["pixel_size"] = (0.2, 0.2)
    l5_config["raster_params"]["ego_center"] = (0.5, 0.5)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, LocalDataManager(None))
    return render_rasterizer


def get_nusc_renderer(dataset_path):
    kwargs = dict(
        desired_data=["val"],
        future_sec=(1.5, 1.5),
        history_sec=(1.0, 1.0),
        data_dirs={
            "nusc_trainval": dataset_path,
            "nusc_mini": dataset_path,
        },
        only_types=[AgentType.VEHICLE],
        agent_interaction_distances=defaultdict(lambda: 30),
        # incl_map=False,
        num_workers=os.cpu_count(),
        desired_dt=0.1
    )

    dataset = UnifiedDataset(**kwargs)

    renderer = NuscRenderer(dataset, raster_size=200, resolution=2)

    return renderer


def get_state_image_l5(rasterizer, ras_pos, ras_yaw):
    state_im = rasterizer.rasterize(
        ras_pos,
        ras_yaw
    )

    raster_from_world = rasterizer.render_context.raster_from_world(
        ras_pos,
        ras_yaw
    )

    return state_im, raster_from_world


class NuscRenderer(object):
    def __init__(self, dataset, raster_size=500, resolution=2):
        self.dataset = dataset
        self.raster_size = raster_size
        self.resolution = resolution
        num_total_scenes = dataset.num_scenes()
        scene_info = dict()
        for i in range(num_total_scenes):
            si = dataset.get_scene(i)
            scene_info[si.name] = si
        self.scene_info = scene_info

    def render(self, ras_pos, ras_yaw, scene_name):
        scene_info = self.scene_info[scene_name]
        cache = SimulationDataFrameCache(
            self.dataset.cache_path,
            scene_info,
            0,
            self.dataset.augmentations,
        )

        patch_data, _, _ = cache.load_map_patch(
            ras_pos[0],
            ras_pos[1],
            self.raster_size,
            self.resolution,
            (0, 0),
            ras_yaw,
            return_rgb=False
        )

        """
        [
                    "lane",
                    "road_segment",
                    "drivable_area",
                    "road_divider",
                    "lane_divider",
                    "ped_crossing",
                    "walkway",
                ]
        """
        state_im = np.ones((self.raster_size, self.raster_size, 3))
        state_im[patch_data[0] > 0] = np.array([200, 211, 213]) / 255.
        state_im[patch_data[1] > 0] = np.array([164, 184, 196]) / 255.
        state_im[patch_data[2] > 0] = np.array([164, 184, 196]) / 255.
        # state_im[patch_data[5] > 0] = np.array([96, 117, 138]) / 255.

        raster_from_agent = np.array([
            [self.resolution, 0, 0.5 * self.raster_size],
            [0, self.resolution, 0.5 * self.raster_size],
            [0, 0, 1]
        ])

        world_from_agent: np.ndarray = np.array(
            [
                [np.cos(ras_yaw), np.sin(ras_yaw), ras_pos[0]],
                [-np.sin(ras_yaw), np.cos(ras_yaw), ras_pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        agent_from_world = np.linalg.inv(world_from_agent)

        raster_from_world = raster_from_agent @ agent_from_world

        return state_im, raster_from_world


def draw_trajectories(ax, trajectories, raster_from_world, linewidth):
    raster_trajs = transform_points(trajectories, raster_from_world)
    for traj in raster_trajs:
        colorline(
            ax,
            traj[..., 0],
            traj[..., 1],
            cmap="viridis",
            linewidth=linewidth,
            zorder=3,
        )

def draw_action_samples(ax, action_samples, raster_from_world, world_from_agent, linewidth,alpha=0.5):
    world_trajs = GeoUtils.batch_nd_transform_points_np(action_samples,world_from_agent[:,np.newaxis])
    raster_trajs = GeoUtils.batch_nd_transform_points_np(world_trajs, raster_from_world[np.newaxis,np.newaxis])
    raster_trajs = TensorUtils.join_dimensions(raster_trajs,0,2)
    interval=5
    for i in range(raster_trajs.shape[0]//interval):

        ax.plot(raster_trajs[i*interval,:,0],raster_trajs[i*interval,:,1],color="m",linewidth=linewidth,linestyle=(0, (3, 1, 1, 1, 1, 1)),zorder=1)
    # ax.scatter(raster_trajs[:,::5,0],raster_trajs[:,::5,1],color="m",s=0.1,marker="D")
    # for traj in raster_trajs:
        
        # colorline(
        #     ax,
        #     traj[..., 0],
        #     traj[..., 1],
        #     cmap="RdPu",
        #     linewidth=linewidth,
        #     alpha=alpha,
        # )

def draw_agent_boxes_plt(ax, pos, yaw, extent, raster_from_agent, outline_color, fill_color):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2))
    for b in boxes_raster:
        rect = patches.Polygon(b, fill=True, color=fill_color, zorder=2)
        rect_border = patches.Polygon(b, fill=False, color="grey", zorder=1, linewidth=0.5)
        ax.add_patch(rect)
        ax.add_patch(rect_border)


def draw_scene_data(ax, scene_name, scene_data, starting_frame, rasterizer, draw_trajectory=True, draw_action_sample = False, focus_agent_id = None, traj_len=200, ras_pos=None, linewidth=3.0):
    t = starting_frame
    if ras_pos is None:
        ras_pos = scene_data["centroid"][0, t]

    if isinstance(rasterizer, NuscRenderer):
        state_im, raster_from_world = rasterizer.render(
            ras_pos=ras_pos,
            # ras_yaw=scene_data["yaw"][0, t],
            ras_yaw=0,
            # ras_yaw=np.pi,
            scene_name=scene_name
        )
        extent_scale = 1.0
    else:
        state_im, raster_from_world = get_state_image_l5(
            rasterizer,
            ras_pos=ras_pos,
            # ras_yaw=scene_data["yaw"][0, t],
            ras_yaw=np.pi
        )
        extent_scale = 1.0
    ax.imshow(state_im)


    
    if draw_action_sample==True and "action_sample_positions" in scene_data:
        if focus_agent_id is not None:
            t0 = t
            while t0>=1:
                if (scene_data["action_sample_positions"][focus_agent_id,t0-1]==scene_data["action_sample_positions"][focus_agent_id,t]).all():
                    t0-=1
                else:
                    break
            draw_action_samples(
                ax,
                action_samples=scene_data["action_sample_positions"][focus_agent_id, t0],
                raster_from_world=raster_from_world,
                world_from_agent = scene_data["world_from_agent"][focus_agent_id,t0],
                linewidth=linewidth*0.1
            )
    if draw_trajectory:
        draw_trajectories(
            ax,
            trajectories=scene_data["centroid"][:, t:t+traj_len],
            raster_from_world=raster_from_world,
            linewidth=linewidth
        )

    draw_agent_boxes_plt(
        ax,
        pos=scene_data["centroid"][:, t],
        yaw=scene_data["yaw"][:, [t]],
        extent=scene_data["extent"][:, t, :2] * extent_scale,
        raster_from_agent=raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    # ax.set_xlim([0, state_im.shape[1] - 140])
    # ax.set_ylim([80, state_im.shape[0] - 130])
    ax.set_xlim([0, state_im.shape[1]])
    ax.set_ylim([0, state_im.shape[0]])
    ax.grid(False)
    ax.axis("off")
    ax.invert_xaxis()


def visualize_scene(rasterizer, h5f, scene_index, starting_frame, output_dir):
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ep_i in range(5):
        ax = axes[ep_i]
        scene_name = "{}_{}".format(scene_index, ep_i)
        if scene_name not in list(h5f.keys()):
            continue
        scene_data = h5f[scene_name]
        draw_scene_data(ax, scene_index, scene_data, starting_frame, rasterizer,draw_action_sample=True,focus_agent_id=[0],traj_len=20)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ffn = os.path.join(output_dir, "{}_t{}.png").format(scene_index, starting_frame)
    plt.savefig(ffn, dpi=400, bbox_inches="tight")
    plt.close()
    del fig, axes
    print("Figure written to {}".format(ffn))


def preprocess(scene_data):
    data = dict()
    for k in scene_data.keys():
        data[k] = scene_data[k][:].copy()
    for i in range(data["yaw"].shape[0]):
        idx = np.where(~np.isnan(data["yaw"][i]))[0]
        data["yaw"][i,idx] = savgol_filter(data["yaw"][i,idx], 11, 3)
    return data


def scene_to_video(rasterizer, h5f, scene_index, output_dir):
    for ep_i in range(5):
        scene_name = "{}_{}".format(scene_index, ep_i)
        if scene_name not in list(h5f.keys()):
            continue
        scene_data = h5f[scene_name]
        scene_data = preprocess(scene_data)
        video_dir = os.path.join(output_dir, scene_name)
        for frame_i in range(scene_data["centroid"].shape[1]):
            fig, ax = plt.subplots()
            draw_scene_data(
                ax,
                scene_index,
                scene_data,
                frame_i,
                rasterizer,
                draw_trajectory=True,
                draw_action_sample=True,
                focus_agent_id=[0],
                traj_len=20,
                linewidth=2.0,
                # ras_pos=scene_data["centroid"][0, 0]
            )

            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            ffn = os.path.join(video_dir, "{:03d}.png").format(frame_i)
            
            plt.savefig(ffn, dpi=400, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            print("Figure written to {}".format(ffn))
            del fig, ax
        writer = imageio.get_writer(os.path.join(video_dir, "{}_anim.mp4".format(scene_name)), fps=10)

        for file in sorted(glob.glob(os.path.join(video_dir,"*.png"))):
            im = imageio.imread(file)
            writer.append_data(im)
        writer.close()


def main(hdf5_path, dataset_path, output_dir, env):
    # SOI_l5kit: [1069, 1090, 4558, ]
    if env == "l5kit":
        rasterizer = get_l5_rasterizer(dataset_path)
        sids = EvaluationConfig().l5kit.eval_scenes
        sids = [1069, 1090, 4558, ]
    else:
        rasterizer = get_nusc_renderer(dataset_path)
        sids = EvaluationConfig().nusc.eval_scenes
        scene_names = list(rasterizer.scene_info.keys())
        sids = [scene_names[si] for si in sids]
        # sids = ["scene-0093","scene-0018", "scene-0095", "scene-0098", "scene-0521", "scene-0523",
        #         "scene-0560", "scene-0627", "scene-0638", "scene-0904"]
        # sids = [ "scene-0017"]

    h5f = h5py.File(hdf5_path, "r")

    # sids = [1069]
    for si in sids:
        visualize_scene(rasterizer, h5f, si, 0, output_dir=output_dir)
        scene_to_video(rasterizer, h5f, si, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=None,
        required=True,
        help="An hdf5 containing the saved rollout info"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations/"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "l5kit"],
        required=True
    )

    args = parser.parse_args()

    main(args.hdf5_path, args.dataset_path, args.output_dir, args.env)
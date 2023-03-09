import numpy as np
from PIL import Image, ImageDraw

from l5kit.geometry import transform_points
from l5kit.rasterization.render_context import RenderContext
from l5kit.configs.config import load_metadata
from trajdata.maps.raster_map import RasterizedMap

from tbsim.utils.tensor_utils import map_ndarray
from tbsim.l5kit.vis_rasterizer import VisualizationRasterizer
from tbsim.utils.geometry_utils import get_box_world_coords_np
import tbsim.utils.tensor_utils as TensorUtils


COLORS = {
    "agent_contour": "#247BA0",
    "agent_fill": "#56B1D8",
    "ego_contour": "#911A12",
    "ego_fill": "#FE5F55",
}


def agent_to_raster_np(pt_tensor, trans_mat):
    pos_raster = transform_points(pt_tensor[None], trans_mat)[0]
    return pos_raster


def draw_actions(
        state_image,
        trans_mat,
        pred_action=None,
        pred_plan=None,
        pred_plan_info=None,
        ego_action_samples=None,
        plan_samples=None,
        action_marker_size=3,
        plan_marker_size=8,
):
    im = Image.fromarray((state_image * 255).astype(np.uint8))
    draw = ImageDraw.Draw(im)

    if pred_action is not None:
        raster_traj = agent_to_raster_np(
            pred_action["positions"].reshape(-1, 2), trans_mat)
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#FE5F55", outline="#911A12")
    if ego_action_samples is not None:
        raster_traj = agent_to_raster_np(
            ego_action_samples["positions"].reshape(-1, 2), trans_mat)
        for point in raster_traj:
            circle = np.hstack([point - action_marker_size, point + action_marker_size])
            draw.ellipse(circle.tolist(), fill="#808080",
                         outline="#911A12")

    if pred_plan is not None:
        pos_raster = agent_to_raster_np(
            pred_plan["positions"][:, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    if plan_samples is not None:
        pos_raster = agent_to_raster_np(
            plan_samples["positions"][0, :, -1], trans_mat)
        for pos in pos_raster:
            circle = np.hstack([pos - plan_marker_size, pos + plan_marker_size])
            draw.ellipse(circle.tolist(), fill="#FF6B35")

    im = np.asarray(im)
    # visualize plan heat map
    if pred_plan_info is not None and "location_map" in pred_plan_info:
        import matplotlib.pyplot as plt

        cm = plt.get_cmap("jet")
        heatmap = pred_plan_info["location_map"][0]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / heatmap.max()
        heatmap = cm(heatmap)

        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap = heatmap.resize(size=(im.shape[1], im.shape[0]))
        heatmap = np.asarray(heatmap)[..., :3]
        padding = np.ones((im.shape[0], 200, 3), dtype=np.uint8) * 255

        composite = heatmap.astype(np.float32) * \
            0.3 + im.astype(np.float32) * 0.7
        composite = composite.astype(np.uint8)
        im = np.concatenate((im, padding, heatmap, padding, composite), axis=1)

    return im


def draw_agent_boxes(image, pos, yaw, extent, raster_from_agent, outline_color, fill_color):
    boxes = get_box_world_coords_np(pos, yaw, extent)
    boxes_raster = transform_points(boxes, raster_from_agent)
    boxes_raster = boxes_raster.reshape((-1, 4, 2)).astype(int)

    im = Image.fromarray((image * 255).astype(np.uint8))
    im_draw = ImageDraw.Draw(im)
    for b in boxes_raster:
        im_draw.polygon(xy=b.reshape(-1).tolist(),
                        outline=outline_color, fill=fill_color)

    im = np.asarray(im).astype(np.float32) / 255.
    return im


def render_state_trajdata(
        batch: dict,
        batch_idx: int,
        action,
) -> np.ndarray:
    pos = batch["history_positions"][batch_idx, -1]
    yaw = batch["history_yaws"][batch_idx, -1]
    extent = batch["extent"][batch_idx, :2]

    image = RasterizedMap.to_img(
        TensorUtils.to_tensor(batch["maps"][batch_idx]),
        [[0], [1], [2]],
    )

    image = draw_agent_boxes(
        image,
        pos=pos[None, :],
        yaw=yaw[None, :],
        extent=extent[None, :],
        raster_from_agent=batch["raster_from_agent"][batch_idx],
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"]
    )

    scene_index = batch["scene_index"][batch_idx]
    agent_scene_index= scene_index == batch["scene_index"]
    agent_scene_index[batch_idx] = 0  # don't plot ego

    neigh_pos = batch["centroid"][agent_scene_index]
    neigh_yaw = batch["yaw"][agent_scene_index]
    neigh_extent = batch["extent"][agent_scene_index, :2]

    if neigh_pos.shape[0] > 0:
        image = draw_agent_boxes(
            image,
            pos=neigh_pos,
            yaw=neigh_yaw[:, None],
            extent=neigh_extent,
            raster_from_agent=batch["raster_from_world"][batch_idx],
            outline_color=COLORS["agent_contour"],
            fill_color=COLORS["agent_fill"]
        )

    plan_info = None
    plan_samples = None
    action_samples = None
    if "plan_info" in action.agents_info:
        plan_info = TensorUtils.map_ndarray(action.agents_info["plan_info"], lambda x: x[[batch_idx]])
    if "plan_samples" in action.agents_info:
        plan_samples = TensorUtils.map_ndarray(action.agents_info["plan_samples"], lambda x: x[[batch_idx]])
    if "action_samples" in action.agents_info:
        action_samples = TensorUtils.map_ndarray(action.agents_info["action_samples"], lambda x: x[[batch_idx]])

    vis_action = TensorUtils.map_ndarray(action.agents.to_dict(), lambda x: x[batch_idx])
    image = draw_actions(
        image,
        trans_mat=batch["raster_from_agent"][batch_idx],
        pred_action=vis_action,
        pred_plan_info=plan_info,
        ego_action_samples=action_samples,
        plan_samples=plan_samples,
        action_marker_size=2,
        plan_marker_size=3
    )
    return image


def get_state_image_with_boxes_l5kit(ego_obs, agents_obs, rasterizer):
    yaw = ego_obs["yaw"]  # set to 0 to fix the video
    state_im = rasterizer.rasterize(
        ego_obs["centroid"],
        yaw
    )

    raster_from_world = rasterizer.render_context.raster_from_world(
        ego_obs["centroid"],
        yaw
    )
    raster_from_agent = raster_from_world @ ego_obs["world_from_agent"]
    
    state_im = draw_agent_boxes(
        state_im,
        agents_obs["centroid"],
        agents_obs["yaw"][:, None],
        agents_obs["extent"][:, :2],
        raster_from_world,
        outline_color=COLORS["agent_contour"],
        fill_color=COLORS["agent_fill"]
    )

    state_im = draw_agent_boxes(
        state_im,
        ego_obs["centroid"][None],
        ego_obs["yaw"][None, None],
        ego_obs["extent"][None, :2],
        raster_from_world,
        outline_color=COLORS["ego_contour"],
        fill_color=COLORS["ego_fill"]
    )

    return state_im, raster_from_agent, raster_from_world


def render_state_l5kit_ego_view(
        rasterizer: VisualizationRasterizer,
        state_obs,
        action,
        step_index,
        dataset_scene_index,
        step_metrics=None,

):

    """Render ego-centric view, possibly with a location heatmap (if using SpatialPlanner)"""
    agent_scene_index = dataset_scene_index == state_obs["agents"]["scene_index"]
    agents_obs = map_ndarray(
        state_obs["agents"], lambda x: x[agent_scene_index])
    ego_scene_index = dataset_scene_index == state_obs["ego"]["scene_index"]
    ego_obs = map_ndarray(state_obs["ego"], lambda x:  x[ego_scene_index][0])

    pred_actions = None
    pred_plan = None
    pred_plan_info = None
    ego_action_samples = None

    if action.ego is not None:
        pred_actions = map_ndarray(
            action.ego.to_dict(), lambda x:  x[ego_scene_index])
        pred_plan = action.ego_info.get("plan", None)
        pred_plan_info = action.ego_info.get("plan_info", None)
        ego_action_samples = action.ego_info.get("action_samples", None)

    if pred_plan is not None:
        pred_plan = map_ndarray(pred_plan, lambda x:  x[ego_scene_index])
        pred_plan_info = map_ndarray(
            pred_plan_info, lambda x:  x[ego_scene_index])
    if ego_action_samples is not None:
        ego_action_samples = map_ndarray(
            ego_action_samples, lambda x:  x[ego_scene_index])

    state_im, raster_from_agent, _ = get_state_image_with_boxes_l5kit(
        ego_obs, agents_obs, rasterizer)

    state_im = draw_actions(
        state_image=state_im,
        trans_mat=raster_from_agent,
        pred_action=pred_actions,
        pred_plan=pred_plan,
        pred_plan_info=pred_plan_info,
        ego_action_samples=ego_action_samples
    )

    return state_im


def render_state_l5kit_agents_view(
        rasterizer: VisualizationRasterizer,
        state_obs,
        action,
        step_index,
        dataset_scene_index,
        num_agent_to_render=1,
        divider_padding_size=0,
        step_metrics=None,
):
    """Render state centered at each agent (including agent). Concatenate each view to width-wise"""
    # get observation by scene
    agent_scene_index = dataset_scene_index == state_obs["agents"]["scene_index"]
    agents_obs = map_ndarray(
        state_obs["agents"], lambda x: x[agent_scene_index])
    ego_scene_index = dataset_scene_index == state_obs["ego"]["scene_index"]
    ego_obs = map_ndarray(state_obs["ego"], lambda x:  x[ego_scene_index])

    # collate ego and agent obs
    all_obs = ego_obs
    if agents_obs is not None:
        for k in ego_obs.keys():
            all_obs[k] = np.concatenate((ego_obs[k], agents_obs[k]), axis=0)

    # collate actions
    ego_action = map_ndarray(action.ego.to_dict(),
                             lambda x:  x[ego_scene_index])
    agents_action = map_ndarray(
        action.agents.to_dict(), lambda x: x[agent_scene_index])
    all_action = dict()
    for k in ego_action:
        all_action[k] = np.concatenate(
            (ego_action[k][None], agents_action[k]), axis=0)

    all_action["positions"] = transform_points(
        all_action["positions"], all_obs["world_from_agent"])

    # collate plans
    ego_plan = map_ndarray(
        action.ego_info["plan"], lambda x:  x[ego_scene_index])
    agents_plan = map_ndarray(
        action.agents_info["plan"], lambda x:  x[agent_scene_index])
    all_plan = dict()
    for k in ego_plan:
        all_plan[k] = np.concatenate(
            (ego_plan[k][None], agents_plan[k]), axis=0)

    all_plan["position"] = transform_points(
        all_plan["positions"], all_obs["world_from_agent"])

    num_agents = min(all_obs["centroid"].shape[0], num_agent_to_render)
    all_ims = []
    for i in range(num_agents):
        agents_inds = np.arange(all_obs["centroid"].shape[0]) != i
        state_im, raster_from_agent, raster_from_world = get_state_image_with_boxes_l5kit(
            ego_obs=map_ndarray(all_obs, lambda x: x[i]),
            agents_obs=map_ndarray(all_obs, lambda x: x[agents_inds]),
            rasterizer=rasterizer
        )
        agent_im = draw_actions(
            state_image=state_im,
            trans_mat=raster_from_world,
            pred_action=all_action,
            pred_plan=all_plan
        )
        padding = np.ones(
            (agent_im.shape[0], divider_padding_size, 3), dtype=np.uint8) * 255
        all_ims.extend([agent_im, padding])

    # pad the rendering to num_agent_to_render in case there are fewer agents
    if num_agents < num_agent_to_render:
        for i in range(num_agent_to_render - num_agents):
            padding = np.ones(
                (all_ims[0].shape[0], all_ims[0].shape[1] +
                 divider_padding_size, 3),
                dtype=np.uint8
            ) * 255
            all_ims.append(padding)

    im = np.concatenate(all_ims, axis=1)  # concatenate everything horizontally
    return im


def build_visualization_rasterizer_l5kit(cfg, dm):
    raster_cfg = cfg["raster_params"]
    dataset_meta_key = raster_cfg["dataset_meta_key"]
    dataset_meta = load_metadata(dm.require(dataset_meta_key))
    world_to_ecef = np.array(dataset_meta["world_to_ecef"], dtype=np.float64)

    render_context = RenderContext(
        raster_size_px=np.array(raster_cfg["raster_size"]),
        pixel_size_m=np.array(raster_cfg["pixel_size"]),
        center_in_raster_ratio=np.array(raster_cfg["ego_center"]),
        set_origin_to_bottom=raster_cfg["set_origin_to_bottom"],
    )

    semantic_map_filepath = dm.require(raster_cfg["semantic_map_key"])
    return VisualizationRasterizer(render_context, semantic_map_filepath, world_to_ecef)

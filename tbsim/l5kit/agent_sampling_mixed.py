from typing import Optional

import numpy as np
from tbsim.l5kit.vectorizer import Vectorizer

from l5kit.data import filter_agents_by_labels, PERCEPTION_LABEL_TO_INDEX
from l5kit.data.filter import filter_agents_by_track_id
from l5kit.geometry import compute_agent_pose, rotation33_as_yaw, transform_points, angular_distance
from l5kit.kinematic import Perturbation
from l5kit.sampling.agent_sampling import (
    compute_agent_velocity,
    get_agent_context,
    get_relative_poses,
)
from l5kit.rasterization import (
    EGO_EXTENT_HEIGHT,
    EGO_EXTENT_LENGTH,
    EGO_EXTENT_WIDTH,
    Rasterizer,
    RenderContext,
)


from l5kit.data.map_api import InterpolationMethod, MapAPI
from l5kit.rasterization.semantic_rasterizer import indices_in_bounds
from l5kit.geometry import transform_points
from tbsim.utils.geometry_utils import batch_proj
from tbsim.utils.tensor_utils import round_2pi
from scipy.interpolate import interp1d


def generate_agent_sample_mixed(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_track_id: Optional[int],
    history_num_frames_ego: int,
    history_num_frames_agents: int,
    future_num_frames: int,
    step_time: float,
    filter_agents_threshold: float,
    vectorizer: Vectorizer,
    rasterizer: Rasterizer,
    render_context: RenderContext,
    timer,
    perturbation: Optional[Perturbation] = None,
    vectorize_lane=False,
    skimp_fn=False,
    rasterize_agents=False,
    vectorize_agents=True,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model with vectorized inputs.
    A deep prediction model takes as input the state of the world in vectorized form,
    and outputs where that agent will be some seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        predict the future of. This agent is centered in the representation and the returned targets are derived from
        their future states.
        history_num_frames_ego (int): Amount of ego history frames to include
        history_num_frames_agents (int): Amount of agent history frames to include
        future_num_frames (int): Amount of future frames to include
        step_time (float): seconds between consecutive steps
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
        to train models that can recover from slight divergence from training set data

    Raises:
        IndexError: An IndexError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict containing e.g. the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask,
        the vectorized input representation features, and (optional) a raster image
    """
    timer.tic("sample")
    with timer.timed("get_agent_context"):
        history_num_frames_max = max(history_num_frames_ego, history_num_frames_agents)
        (
            history_frames,
            future_frames,
            history_agents,
            future_agents,
            history_tl_faces,
            future_tl_faces,
        ) = get_agent_context(
            state_index,
            frames,
            agents,
            tl_faces,
            history_num_frames_max,
            future_num_frames,
        )


    if perturbation is not None and len(future_frames) == future_num_frames:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # State you want to predict the future of.
    cur_frame = history_frames[0]
    cur_agents = history_agents[0]


    if selected_track_id is None:
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray(
            (EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT)
        )
        agent_type_idx = PERCEPTION_LABEL_TO_INDEX["PERCEPTION_LABEL_CAR"]
        selected_agent = None
    else:
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        try:
            agent = filter_agents_by_track_id(
                filter_agents_by_labels(cur_agents, filter_agents_threshold),
                selected_track_id,
            )[0]
        except IndexError:
            agent = filter_agents_by_track_id(cur_agents,selected_track_id)[0]
            # raise ValueError(
            #     f" track_id {selected_track_id} not in frame or below threshold"
            # )
        agent_centroid_m = agent["centroid"]
        agent_yaw_rad = float(agent["yaw"])
        agent_extent_m = agent["extent"]
        agent_type_idx = np.argmax(agent["label_probabilities"])
        selected_agent = agent

    with timer.timed("rasterize"):
        rasterizer_out = dict()
        if not skimp_fn():
            rasterizer_out["image"] = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)

    raster_from_world = render_context.raster_from_world(
        agent_centroid_m, agent_yaw_rad
    )

    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    agent_from_world = np.linalg.inv(world_from_agent)

    raster_from_agent = raster_from_world @ world_from_agent
    agent_from_raster = np.linalg.inv(raster_from_agent)

    with timer.timed("get_future_history"):
        (
            future_coords_offset,
            future_yaws_offset,
            future_extents,
            future_availability,
        ) = get_relative_poses(
            future_num_frames,
            future_frames,
            selected_track_id,
            future_agents,
            agent_from_world,
            agent_yaw_rad,
        )


        # For vectorized version we require both ego and agent history to be a Tensor of same length
        # => fetch history_num_frames_max for both, and later zero out frames exceeding the set history length.
        # Use history_num_frames_max + 1 because it also includes the current frame.
        (
            history_coords_offset,
            history_yaws_offset,
            history_extents,
            history_availability,
        ) = get_relative_poses(
            history_num_frames_max + 1,
            history_frames,
            selected_track_id,
            history_agents,
            agent_from_world,
            agent_yaw_rad,
        )

    history_coords_offset[history_num_frames_ego + 1 :] *= 0
    history_yaws_offset[history_num_frames_ego + 1 :] *= 0
    history_extents[history_num_frames_ego + 1 :] *= 0
    history_availability[history_num_frames_ego + 1 :] *= 0

    history_vels_mps, future_vels_mps = compute_agent_velocity(
        history_coords_offset, future_coords_offset, step_time
    )
    frame_info = {
        "extent": agent_extent_m,
        "type": agent_type_idx,
        "raster_from_agent": raster_from_agent,
        "agent_from_raster": agent_from_raster,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "target_positions": future_coords_offset,
        "target_yaws": future_yaws_offset,
        "target_extents": future_extents,
        "target_availabilities": future_availability.astype(bool),
        "history_positions": history_coords_offset,
        "history_yaws": history_yaws_offset,
        "history_extents": history_extents,
        "history_availabilities": history_availability.astype(bool),
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm(future_vels_mps[0]),
        "ego_translation": np.array(cur_frame["ego_translation"]),
        "curr_speed": np.linalg.norm(history_vels_mps[0]),
    }

    with timer.timed("vectorize"):
        if not skimp_fn() and vectorize_agents:
            vectorized_features = vectorizer.vectorize(
                selected_track_id,
                agent_centroid_m,
                agent_yaw_rad,
                agent_from_world,
                history_frames,
                history_agents,
                history_tl_faces,
                history_coords_offset,
                history_yaws_offset,
                history_availability,
                future_frames,
                future_agents,
            )
            if rasterize_agents:
                num_agent = vectorized_features["all_other_agents_history_availability"].shape[0]
                agent_index = np.where(vectorized_features["all_other_agents_history_availability"][:, 0]\
                                    & vectorized_features["all_other_agents_future_availability"].all(axis=-1)\
                                    & (vectorized_features["all_other_agents_types"] >= 3)\
                                    & (vectorized_features["all_other_agents_types"] <= 13))[0].tolist()
                agent_raster_availability = np.zeros(num_agent,dtype=bool)
                agent_raster_availability[agent_index]=True
                other_agents_image = np.zeros([num_agent,*rasterizer_out["image"].shape],dtype=np.float32)
                other_agents_raster_from_world = np.tile(np.zeros([3,3],dtype=np.float32),[num_agent,1,1])
                other_agents_agent_from_raster = np.tile(np.zeros([3,3],dtype=np.float32),[num_agent,1,1])
                other_agents_world_from_agent = np.tile(np.zeros([3,3],dtype=np.float32),[num_agent,1,1])
                other_agents_raster_from_agent = np.tile(np.zeros([3,3],dtype=np.float32),[num_agent,1,1])
                other_agents_agent_from_world = np.tile(np.zeros([3,3],dtype=np.float32),[num_agent,1,1])
                for idx in agent_index:
                    track_id = vectorized_features["all_other_agents_track_id"][idx]
                    agent = filter_agents_by_track_id(cur_agents,track_id)[0]
                    other_agents_image[idx] = rasterizer.rasterize(history_frames, history_agents, history_tl_faces, agent)
                    agent_centroid_m = agent["centroid"]
                    agent_yaw_rad = float(agent["yaw"])
                    raster_from_world = render_context.raster_from_world(
                        agent_centroid_m, agent_yaw_rad
                    )
                    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
                    agent_from_world = np.linalg.inv(world_from_agent)
                    raster_from_agent = raster_from_world @ world_from_agent
                    agent_from_raster = np.linalg.inv(raster_from_agent)
                    other_agents_raster_from_world[idx] = raster_from_world
                    other_agents_agent_from_raster[idx] = agent_from_raster
                    other_agents_world_from_agent[idx] = world_from_agent
                    other_agents_raster_from_agent[idx] = raster_from_agent
                    other_agents_agent_from_world[idx] = agent_from_world
                vectorized_features["other_agents_image"] = other_agents_image
                vectorized_features["agent_raster_availability"] = agent_raster_availability
                vectorized_features["other_agents_raster_from_world"]=other_agents_raster_from_world
                vectorized_features["other_agents_agent_from_raster"]=other_agents_agent_from_raster
                vectorized_features["other_agents_world_from_agent"]=other_agents_world_from_agent
                vectorized_features["other_agents_raster_from_agent"]=other_agents_raster_from_agent
                vectorized_features["other_agents_agent_from_world"] =other_agents_agent_from_world
        else:
            vectorized_features = dict()
    if not skimp_fn() and vectorize_agents and vectorize_lane:
        other_agents_idx = np.where(
            vectorized_features["all_other_agents_history_availability"][:, 0]
            & (vectorized_features["all_other_agents_types"] >= 3)
            & (vectorized_features["all_other_agents_types"] <= 13)
        )[0]
        available_other_pos = vectorized_features["all_other_agents_history_positions"][
            other_agents_idx, 0
        ]
        available_other_yaw = vectorized_features["all_other_agents_history_yaws"][
            other_agents_idx, 0
        ]
        local_pos = np.vstack((np.zeros([1, 2]), available_other_pos))
        local_yaw = np.vstack((np.zeros([1, 1]), available_other_yaw))
        world_pos = transform_points(local_pos, world_from_agent)
        world_yaw = (local_yaw + agent_yaw_rad + np.pi) % (2 * np.pi) - np.pi
    
        agent_lanes = get_lane_info(
            agent_yaw_rad,
            vectorizer,
            world_pos,
            world_yaw,
            local_pos,
            local_yaw,
            world_from_agent,
            agent_from_world,
        )
        ego_lanes = agent_lanes[0]
        all_other_agents_lanes = np.zeros(
            [
                vectorized_features["all_other_agents_history_positions"].shape[0],
                *agent_lanes.shape[1:],
            ]
        )
        all_other_agents_lanes[other_agents_idx] = agent_lanes[1:]
        frame_info["ego_lanes"] = ego_lanes
        frame_info["all_other_agents_lanes"] = all_other_agents_lanes
    timer.toc("sample")

    return {**frame_info, **vectorized_features, **rasterizer_out}


def get_lane_info(
    yaw,
    vectorizer,
    world_pos,
    world_yaw,
    local_pos,
    local_yaw,
    world_from_agent,
    agent_from_world,
):
    MAX_LANES = vectorizer.lane_cfg_params["max_num_lanes"]
    MAX_POINTS_LANES = vectorizer.lane_cfg_params["max_points_per_lane"]
    # MAX_POINTS_CW = vectorizer.lane_cfg_params["max_points_per_crosswalk"]

    MAX_LANE_DISTANCE = vectorizer.lane_cfg_params["max_retrieval_distance_m"]
    INTERP_METHOD = (
        InterpolationMethod.INTER_ENSURE_LEN
    )  # split lane polyline by fixed number of points
    STEP_INTERPOLATION = MAX_POINTS_LANES  # number of points along lane
    MAX_CROSSWALKS = vectorizer.lane_cfg_params["max_num_crosswalks"]

    # lanes_points = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES, 2), dtype=np.float32)
    # lanes_availabilities = np.zeros((MAX_LANES * 2, MAX_POINTS_LANES), dtype=np.float32)

    # lanes_mid_points = np.zeros((MAX_LANES, MAX_POINTS_LANES, 2), dtype=np.float32)
    # lanes_mid_availabilities = np.zeros((MAX_LANES, MAX_POINTS_LANES), dtype=np.float32)
    # lanes_tl_feature = np.zeros((MAX_LANES, MAX_POINTS_LANES, 1), dtype=np.float32)

    # 8505 x 2 x 2
    lanes_bounds = vectorizer.mapAPI.bounds_info["lanes"]["bounds"]

    # filter first by bounds and then by distance, so that we always take the closest lanes

    N_agent = world_pos.shape[0]
    curr_lane = [None] * N_agent
    left_lane = [None] * N_agent
    right_lane = [None] * N_agent
    len_curr = [None] * N_agent
    len_left = [None] * N_agent
    len_right = [None] * N_agent
    dx_curr = [None] * N_agent
    dx_left = [None] * N_agent
    dx_right = [None] * N_agent
    agent_lanes = np.zeros([N_agent, 3, MAX_POINTS_LANES, 4], dtype=np.float32)
    lanes_rec = dict()
    interp_step_size = vectorizer.lane_cfg_params["lane_interp_step_size"]
    interp_steps = interp_step_size * np.arange(1, MAX_POINTS_LANES + 1)
    for i in range(N_agent):

        lanes_indices = indices_in_bounds(world_pos[i], lanes_bounds, 10)
        distances = list()
        for k in range(lanes_indices.shape[0]):
            lane_idx = lanes_indices[k]
            if lane_idx in lanes_rec:
                lane = lanes_rec[lane_idx]
            else:
                lane_id = vectorizer.mapAPI.bounds_info["lanes"]["ids"][lane_idx]

                lane = vectorizer.mapAPI.get_lane_coords(lane_id)
                lanes_rec[lane_idx] = lane

                if (
                    lane["xyz_right"].shape[0] != lane["xyz_left"].shape[0]
                    or "xyz_midlane" not in lane
                ):
                    lane = vectorizer.mapAPI.get_lane_as_interpolation(
                        lane_id, STEP_INTERPOLATION, INTERP_METHOD
                    )
                dx = lane["xyz_right"] - lane["xyz_left"]
                lane_psi = round_2pi(np.arctan2(dx[:, 1], dx[:, 0]) + np.pi / 2)

                lane["psi"] = lane_psi

            lane_dist = np.linalg.norm(
                lane["xyz_midlane"][:, :2] - world_pos[i], axis=-1
            )
            distances.append(np.min(lane_dist))
            if distances[-1] < 30.0:

                lane_pts = np.hstack(
                    (lane["xyz_midlane"][:, :2], lane["psi"].reshape(-1, 1))
                )
                x = np.hstack((world_pos[i], world_yaw[i]))
                delta_x, delta_y, dpsi = batch_proj(x, lane_pts)
                min_dy = delta_y[abs(delta_y).argmin()]
                len_cand = -delta_x[-1]
                if abs(min_dy) < 1.5 and abs(dpsi) < np.pi / 2:
                    if curr_lane[i] is None or len_curr[i] < len_cand:
                        curr_lane[i] = lane
                        dx_curr[i] = -delta_x
                        len_curr[i] = len_cand

                elif min_dy <= -1.5 and min_dy > -8 and abs(dpsi) < 0.75*np.pi:
                    if right_lane[i] is None or len_right[i] < len_cand:
                        right_lane[i] = lane
                        dx_right[i] = -delta_x
                        len_right[i] = len_cand

                elif min_dy >= 1.5 and min_dy < 8 and abs(dpsi) < 0.75*np.pi:
                    if left_lane[i] is None or len_left[i] < len_cand:
                        left_lane[i] = lane
                        dx_left[i] = -delta_x
                        len_left[i] = len_cand

        if curr_lane[i] is not None:
            lane_center = curr_lane[i]["xyz_midlane"][:, :2]

            lane_center = transform_points(lane_center, agent_from_world) - local_pos[i]
            lane_yaw = curr_lane[i]["psi"] - yaw

            f = interp1d(
                dx_curr[i],
                np.hstack(
                    (
                        lane_center,
                        np.cos(lane_yaw).reshape(-1, 1),
                        np.sin(lane_yaw).reshape(-1, 1),
                    )
                ),
                fill_value="extrapolate",
                assume_sorted=True,
                axis=0,
            )
            agent_lanes[i, 0] = f(interp_steps)
        if left_lane[i] is not None:
            lane_center = left_lane[i]["xyz_midlane"][:, :2]

            lane_center = transform_points(lane_center, agent_from_world) - local_pos[i]
            lane_yaw = left_lane[i]["psi"] - yaw

            f = interp1d(
                dx_left[i],
                np.hstack(
                    (
                        lane_center,
                        np.cos(lane_yaw).reshape(-1, 1),
                        np.sin(lane_yaw).reshape(-1, 1),
                    )
                ),
                fill_value="extrapolate",
                assume_sorted=True,
                axis=0,
            )
            agent_lanes[i, 1] = f(interp_steps)
        if right_lane[i] is not None:
            lane_center = right_lane[i]["xyz_midlane"][:, :2]

            lane_center = transform_points(lane_center, agent_from_world) - local_pos[i]
            lane_yaw = right_lane[i]["psi"] - yaw

            f = interp1d(
                dx_right[i],
                np.hstack(
                    (
                        lane_center,
                        np.cos(lane_yaw).reshape(-1, 1),
                        np.sin(lane_yaw).reshape(-1, 1),
                    )
                ),
                fill_value="extrapolate",
                assume_sorted=True,
                axis=0,
            )
            agent_lanes[i, 2] = f(interp_steps)
    return agent_lanes
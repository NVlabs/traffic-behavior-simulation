import bisect
import warnings
from functools import partial
from typing import Callable, Optional
from pathlib import Path
from zarr import convenience
from functools import partial
from multiprocessing import cpu_count, Pool
import zarr
from tqdm import tqdm
from collections import Counter
import pprint

import numpy as np
from torch.utils.data import Dataset

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes, get_agents_slice_from_frames
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.dataset import EgoDataset
from l5kit.dataset.select_agents import select_agents, get_valid_agents, TH_DISTANCE_AV, TH_YAW_DEGREE, TH_EXTENT_RATIO
from tbsim.l5kit.vectorizer import Vectorizer
from tbsim.l5kit.agent_sampling_mixed import generate_agent_sample_mixed
from tbsim.utils.timer import Timers

from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed

# WARNING: changing these values impact the number of instances selected for both train and inference!
MIN_FRAME_HISTORY = 10  # minimum number of frames an agents must have in the past to be picked
MIN_FRAME_FUTURE = 1  # minimum number of frames an agents must have in the future to be picked


def select_agents_np(
        zarr_dataset: ChunkedDataset,
        th_agent_prob: float,
        th_yaw_degree: float,
        th_extent_ratio: float,
        th_distance_av: float,
) -> np.ndarray:
    """
    Filter agents from zarr INPUT_FOLDER according to multiple thresholds and store a boolean array of the same shape.
    """
    frame_index_intervals = zarr_dataset.scenes["frame_index_interval"]

    # build a partial with all args except the first one (will be passed by threads)
    get_valid_agents_partial = partial(
        get_valid_agents,
        dataset=zarr_dataset,
        th_agent_filter_probability_threshold=th_agent_prob,
        th_yaw_degree=th_yaw_degree,
        th_extent_ratio=th_extent_ratio,
        th_distance_av=th_distance_av,
    )

    report: Counter = Counter()

    agents_mask = np.zeros((len(zarr_dataset.agents), 2), dtype=np.uint32)

    print("starting pool...")
    with Pool(cpu_count()) as pool:
        tasks = tqdm(enumerate(pool.imap_unordered(get_valid_agents_partial, frame_index_intervals)))
        for idx, (mask, count, agents_range) in tasks:
            report += count
            agents_mask[agents_range[0]: agents_range[1]] = mask
            tasks.set_description(f"{idx + 1}/{len(frame_index_intervals)}")
        print("collecting results..")

    agents_cfg = {
        "th_agent_filter_probability_threshold": th_agent_prob,
        "th_yaw_degree": th_yaw_degree,
        "th_extent_ratio": th_extent_ratio,
        "th_distance_av": th_distance_av,
    }
    # print report
    pp = pprint.PrettyPrinter(indent=4)
    print(f"start report for {zarr_dataset.path}")
    pp.pprint({**agents_cfg, **report})

    return agents_mask


class AgentDataset(EgoDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            read_cached_mask: bool = True,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        assert perturbation is None, "AgentDataset does not support perturbation (yet)"

        super(AgentDataset, self).__init__(cfg, zarr_dataset, rasterizer, perturbation)

        # store the valid agents indices (N_valid_agents,)
        if agents_mask is None:
            if read_cached_mask:
                agents_mask = self.load_agents_mask()
            else:
                agents_mask = select_agents_np(
                    zarr_dataset,
                    th_agent_prob=cfg["raster_params"]["filter_agents_threshold"],
                    th_yaw_degree=TH_YAW_DEGREE,
                    th_extent_ratio=TH_EXTENT_RATIO,
                    th_distance_av=TH_DISTANCE_AV,
                )
            past_mask = agents_mask[:, 0] >= min_frame_history
            future_mask = agents_mask[:, 1] >= min_frame_future
            agents_mask = past_mask * future_mask

        self.agents_indices = np.nonzero(agents_mask)[0]

        # store an array where valid indices have increasing numbers and the rest is -1 (N_total_agents,)
        self.mask_indices = agents_mask.copy().astype(int)
        self.mask_indices[self.mask_indices == 0] = -1
        self.mask_indices[self.mask_indices == 1] = np.arange(0, np.sum(agents_mask))
        # this will be used to get the frame idx from the agent idx
        self.cumulative_sizes_agents = self.dataset.frames["agent_index_interval"][:, 1]
        self.agents_mask = agents_mask

    def load_agents_mask(self) -> np.ndarray:
        """
        Loads a boolean mask of the agent availability stored into the zarr. Performs some sanity check against cfg.
        Returns: a boolean mask of the same length of the dataset agents
        """
        agent_prob = self.cfg["raster_params"]["filter_agents_threshold"]

        agents_mask_path = Path(self.dataset.path) / f"agents_mask/{agent_prob}"
        if not agents_mask_path.exists():  # don't check in root but check for the path
            warnings.warn(
                f"cannot find the right config in {self.dataset.path},\n"
                f"your cfg has loaded filter_agents_threshold={agent_prob};\n"
                "but that value doesn't have a match among the agents_mask in the zarr\n"
                "Mask will now be generated for that parameter.",
                RuntimeWarning,
                stacklevel=2,
            )

            select_agents(
                self.dataset,
                agent_prob,
                th_yaw_degree=TH_YAW_DEGREE,
                th_extent_ratio=TH_EXTENT_RATIO,
                th_distance_av=TH_DISTANCE_AV,
            )

        agents_mask = convenience.load(str(agents_mask_path))  # note (lberg): this doesn't update root
        return agents_mask


    def __len__(self) -> int:
        """
        length of the available and reliable agents (filtered using the mask)
        Returns: the length of the dataset

        """
        return len(self.agents_indices)

    def __getitem__(self, index: int) -> dict:
        """
        Differs from parent by iterating on agents and not AV.
        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        index = self.agents_indices[index]
        track_id = self.dataset.agents[index]["track_id"]
        frame_index = bisect.bisect_right(self.cumulative_sizes_agents, index)
        scene_index = bisect.bisect_right(self.cumulative_sizes, frame_index)

        if scene_index == 0:
            state_index = frame_index
        else:
            state_index = frame_index - self.cumulative_sizes[scene_index - 1]
        data = self.get_frame(scene_index, state_index, track_id=track_id)
        if "other_agents_image" in data:
            data["other_agents_image"] = data["other_agents_image"].transpose(0,3,1,2)
        return data

    def get_scene_dataset(self, scene_index: int) -> "AgentDataset":
        """
        Differs from parent only in the return type.
        Instead of doing everything from scratch, we rely on super call and fix the agents_mask
        """

        new_dataset = super(AgentDataset, self).get_scene_dataset(scene_index).dataset

        # filter agents_bool values
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        # ASSUMPTION: all agents_index are consecutive
        start_index = self.dataset.frames[frame_interval[0]]["agent_index_interval"][0]
        end_index = self.dataset.frames[frame_interval[1] - 1]["agent_index_interval"][1]
        agents_mask = self.agents_mask[start_index:end_index].copy()

        return AgentDataset(
            self.cfg, new_dataset, self.rasterizer, self.perturbation, agents_mask  # overwrite the loaded one
        )

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        frame_slice = get_frames_slice_from_scenes(scenes[scene_idx])
        agent_slice = get_agents_slice_from_frames(*self.dataset.frames[frame_slice][[0, -1]])

        mask_valid_indices = (self.agents_indices >= agent_slice.start) * (self.agents_indices < agent_slice.stop)
        indices = np.nonzero(mask_valid_indices)[0]
        return indices

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. Here __getitem__ iterate over valid agents indices.
        This means ``__getitem__(0)`` matches the first valid agent in the dataset.
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        assert frame_idx < len(self.dataset.frames), f"frame_idx {frame_idx} is over len {len(self.dataset.frames)}"

        # avoid using `get_agents_slice_from_frames` as it hits the disk
        agent_start = self.cumulative_sizes_agents[frame_idx - 1] if frame_idx > 0 else 0
        agent_end = self.cumulative_sizes_agents[frame_idx]
        # slice using frame boundaries and take only valid indices
        mask_idx = self.mask_indices[agent_start:agent_end]
        indices = mask_idx[mask_idx != -1]
        return indices


class AgentDatasetMixed(AgentDataset):
    def __init__(
            self,
            cfg: dict,
            zarr_dataset: ChunkedDataset,
            vectorizer: Vectorizer,
            rasterizer: Rasterizer,
            perturbation: Optional[Perturbation] = None,
            read_cached_mask: bool = True,
            agents_mask: Optional[np.ndarray] = None,
            min_frame_history: int = MIN_FRAME_HISTORY,
            min_frame_future: int = MIN_FRAME_FUTURE,
    ):
        self.vectorizer = vectorizer
        self._skimp = False
        self.timer = Timers()

        super(AgentDatasetMixed, self).__init__(
            cfg=cfg,
            zarr_dataset=zarr_dataset,
            rasterizer=rasterizer,
            perturbation=perturbation,
            read_cached_mask=read_cached_mask,
            agents_mask=agents_mask,
            min_frame_history=min_frame_history,
            min_frame_future=min_frame_future
        )

    def set_skimp(self, skimp):
        self._skimp = skimp

    def is_skimp(self):
        return self._skimp

    def _get_sample_function(self) -> Callable[..., dict]:
        render_context = RenderContext(
            raster_size_px=np.array(self.cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(self.cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(self.cfg["raster_params"]["ego_center"]),
            set_origin_to_bottom=self.cfg["raster_params"]["set_origin_to_bottom"],
        )
        return partial(
            generate_agent_sample_mixed,
            render_context=render_context,
            history_num_frames_ego=self.cfg["model_params"]["history_num_frames_ego"],
            history_num_frames_agents=self.cfg["model_params"][
                "history_num_frames_agents"
            ],
            future_num_frames=self.cfg["model_params"]["future_num_frames"],
            step_time=self.cfg["model_params"]["step_time"],
            filter_agents_threshold=self.cfg["raster_params"][
                "filter_agents_threshold"
            ],
            timer=self.timer,
            perturbation=self.perturbation,
            vectorizer=self.vectorizer,
            rasterizer=self.rasterizer,
            skimp_fn=self.is_skimp,
            vectorize_lane=self.cfg["data_generation_params"]["vectorize_lane"],
            rasterize_agents = self.cfg["data_generation_params"].get("rasterize_agents", False),
            vectorize_agents = self.cfg["data_generation_params"].get("vectorize_agents", True),
        )

    def get_scene_dataset(self, scene_index: int) -> "AgentDatasetMixed":
        """
        Differs from parent only in the return type.
        Instead of doing everything from scratch, we rely on super call and fix the agents_mask
        """

        new_dataset = super(AgentDataset, self).get_scene_dataset(scene_index).dataset

        # filter agents_bool values
        frame_interval = self.dataset.scenes[scene_index]["frame_index_interval"]
        # ASSUMPTION: all agents_index are consecutive
        start_index = self.dataset.frames[frame_interval[0]]["agent_index_interval"][0]
        end_index = self.dataset.frames[frame_interval[1] - 1]["agent_index_interval"][1]
        agents_mask = self.agents_mask[start_index:end_index].copy()

        return AgentDatasetMixed(
            self.cfg, new_dataset, self.vectorizer, self.rasterizer, self.perturbation, agents_mask  # overwrite the loaded one
        )


import bisect
from functools import partial
from typing import Callable, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from l5kit.data import ChunkedDataset, get_frames_slice_from_scenes
from l5kit.dataset.utils import convert_str_to_fixed_length_tensor
from l5kit.kinematic import Perturbation
from l5kit.rasterization import Rasterizer, RenderContext
from l5kit.sampling.agent_sampling import generate_agent_sample
from l5kit.sampling.agent_sampling_vectorized import generate_agent_sample_vectorized
from tbsim.l5kit.agent_sampling_mixed import generate_agent_sample_mixed
from tbsim.l5kit.vectorizer import Vectorizer
from l5kit.dataset.ego import BaseEgoDataset
from tbsim.utils.timer import Timers


class EgoDatasetMixed(BaseEgoDataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        vectorizer: Vectorizer,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNNs with vectorized input

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            vectorizer (Vectorizer): a object that supports vectorization around an AV
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
        None if not desired
        """
        self.perturbation = perturbation
        self.vectorizer = vectorizer
        self.rasterizer = rasterizer
        self.timer = Timers()
        self._skimp = False
        super().__init__(cfg, zarr_dataset)

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

    def get_scene_dataset(self, scene_index: int) -> "EgoDatasetMixed":
        dataset = self.dataset.get_scene_dataset(scene_index)
        return EgoDatasetMixed(
            self.cfg,
            dataset,
            self.vectorizer,
            self.rasterizer,
            self.perturbation,
        )

    def get_frame(
            self, scene_index: int, state_index: int, track_id: Optional[int] = None
    ) -> dict:
        with self.timer.timed("get_frame"):
            data = super().get_frame(scene_index, state_index, track_id=track_id)
        # TODO (@lberg): this should not be here but in the rasterizer
        if "image" in data:
            data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        if "other_agents_image" in data:
            data["other_agents_image"] = data["other_agents_image"].transpose(0,3,1,2)
        return data


class EgoReplayBufferMixed(Dataset):
    """A Dataset class object for wrapping environment interaction episodes"""
    def __init__(
            self,
            cfg,
            vectorizer: Vectorizer,
            rasterizer: Rasterizer,
            capacity=None,
            perturbation: Perturbation = None,
    ):
        super(EgoReplayBufferMixed, self).__init__()
        self.cfg = cfg
        self.dataset = dict()
        self._capacity = capacity
        self._active_scenes = []

        self.perturbation = perturbation
        self.vectorizer = vectorizer
        self.rasterizer = rasterizer

        self.sample_function = self._get_sample_function()

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
            perturbation=self.perturbation,
            vectorizer=self.vectorizer,
            rasterizer=self.rasterizer,
            vectorize_lane=self.cfg["data_generation_params"]["vectorize_lane"],
            rasterize_agents = self.cfg["data_generation_params"]["rasterize_agents"],
        )

    def append_experience(self, episodes_data: List):
        """
        Append list of episodic experience
        Args:
            episodes_data (list): a list of episodic experiences

        """
        self._active_scenes.extend([d[0] for d in episodes_data])
        for si, ds in episodes_data:
            self.dataset[si] = ds

        if self._capacity is not None and len(self._active_scenes) > self._capacity:
            n_to_remove = len(self._active_scenes) - self._capacity
            for si in self._active_scenes[:n_to_remove]:
                self.dataset.pop(si)
            self._active_scenes = self._active_scenes[n_to_remove:]

    def _get_scene_indices(self):
        fi = dict()
        ind = 0
        for si in self._active_scenes:
            fl = len(self.dataset[si].frames)
            fi[si] = (ind, ind + fl)
            ind += fl
        return fi

    def _get_scene_by_index(self, index):
        fi = self._get_scene_indices()
        for si, (start, end) in fi.items():
            if start <= index < end:
                return si, index - start
        raise IndexError("index {} is out of range".format(index))

    def __len__(self):
        return self._get_scene_indices()[self._active_scenes[-1]][1]

    def __getitem__(self, index):
        scene_index, state_index = self._get_scene_by_index(index)
        dataset = self.dataset[scene_index]
        tl_faces = dataset.tl_faces
        if self.cfg["raster_params"]["disable_traffic_light_faces"]:
            tl_faces = np.empty(0, dtype= dataset.tl_faces.dtype)
        data = self.sample_function(
            state_index,
            dataset.frames,
            dataset.agents,
            tl_faces,
            selected_track_id=None
        )
        data["image"] = data["image"].transpose(2, 0, 1)

        # add information only, so that all data keys are always preserved
        data["scene_index"] = scene_index
        data["track_id"] = np.int64(-1)  # always a number to avoid crashing torch
        return data


class ExperienceIterableWrapper(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self._should_update_indices = False
        self._indices = None
        self._rnd = None
        self._curr_index = 0

    def _update_indices(self):
        self._indices = np.arange(len(self.dataset))
        self._rnd.shuffle(self._indices)
        self._curr_index = 0

    def append_experience(self, episodes_data: List):
        self._should_update_indices = True
        self.dataset.append_experience(episodes_data)

    def __iter__(self):
        while True:
            if self._rnd is None:
                winfo = torch.utils.data.get_worker_info()
                if winfo is not None:
                    seed = winfo.id
                else:
                    seed = 0
                self._rnd = np.random.RandomState(seed=seed)
            if self._should_update_indices:
                self._update_indices()
                self._should_update_indices = False
            yield self.dataset[self._curr_index]
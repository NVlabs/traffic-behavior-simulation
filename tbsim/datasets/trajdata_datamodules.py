import os
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tbsim.configs.base import TrainConfig

from trajdata import AgentBatch, AgentType, UnifiedDataset


class UnifiedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, train_config: TrainConfig):
        super(UnifiedDataModule, self).__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
        # TODO: better way to figure out channel size?
        return dict(
            image=(3 + self._data_config.history_num_frames + 1,  # semantic map + num_history + current
                   self._data_config.raster_size,
                   self._data_config.raster_size),
            static=(3,self._data_config.raster_size,self._data_config.raster_size),
            dynamic=(self._data_config.history_num_frames + 1,self._data_config.raster_size,self._data_config.raster_size)

        )

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        kwargs = dict(
            centric = data_cfg.centric,
            desired_data=[data_cfg.trajdata_source_train],
            desired_dt=data_cfg.step_time,
            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            data_dirs={
                data_cfg.trajdata_source_root: data_cfg.dataset_path,
            },
            only_types=[AgentType.VEHICLE],
            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=True,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": False,
                "offset_frac_xy": data_cfg.raster_center,
                "original_format": True,
            },
            cache_location="/raid/local_cache" if self._train_config.on_ngc else "~/.unified_data_cache",
            verbose=False,
            max_agent_num = 1+data_cfg.other_agents_num,
            # max_neighbor_num = data_cfg.other_agents_num,
            num_workers=os.cpu_count(),
            # ego_only = self._train_config.ego_only,
        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = [data_cfg.trajdata_source_valid]
        kwargs["rebuild_cache"] = False
        self.valid_dataset = UnifiedDataset(**kwargs)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers=self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True if self._train_config.training.num_data_workers>0 else False
            
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=True,
            batch_size=self._train_config.validation.batch_size,
            num_workers=self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True if self._train_config.validation.num_data_workers>0 else False
        )

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass

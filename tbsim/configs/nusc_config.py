import math

from tbsim.configs.base import TrainConfig, EnvConfig, AlgoConfig

MAX_POINTS_LANE = 5


class NuscTrainConfig(TrainConfig):
    def __init__(self):
        super(NuscTrainConfig, self).__init__()

        self.trajdata_source_train = "train"
        self.trajdata_source_valid = "val"
        self.trajdata_source_root = "nusc_trainval"

        self.dataset_path = "SET-THIS-THROUGH-TRAIN-SCRIPT-ARGS"
        self.datamodule_class = "UnifiedDataModule"
        self.ego_only=False

        self.rollout.enabled = False
        self.rollout.save_video = True
        self.rollout.every_n_steps = 5000

        # training config
        self.training.batch_size = 100
        self.training.num_steps = 200000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 1000
        self.save.best_k = 10
        

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 6
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 50


class NuscEnvConfig(EnvConfig):
    def __init__(self):
        super(NuscEnvConfig, self).__init__()

        self.name = "nusc_trainval"

        # raster image size [pixels]
        self.rasterizer.raster_size = 224

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 0.5

        # where the agent is on the map, (0.0, 0.0) is the center
        # WARNING: this should not be changed before resolving TODO in parse_trajdata_batch() in trajdata_utils.py
        self.rasterizer.ego_center = (-0.5, 0.0)

        # maximum number of agents to consider during training
        self.data_generation_params.other_agents_num = 20

        self.data_generation_params.max_agents_distance = 30

        # correct for yaw (zero-out delta yaw) when speed is lower than this threshold
        self.data_generation_params.yaw_correction_speed = 1.0

        self.simulation.distance_th_close = 30

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0

        # whether to get lane information
        self.simulation.vectorize_lane = "ego"

        # whether include neighbor map patches
        self.incl_neighbor_map = False
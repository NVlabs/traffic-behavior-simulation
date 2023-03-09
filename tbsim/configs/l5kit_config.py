from tbsim.configs.base import TrainConfig, EnvConfig


class L5KitTrainConfig(TrainConfig):
    def __init__(self):
        super(L5KitTrainConfig, self).__init__()

        self.dataset_path = "/home/yuxiaoc/repos/l5kit/prediction-dataset"
        self.dataset_valid_key = "scenes/validate.zarr"
        self.dataset_train_key = "scenes/train.zarr"
        self.dataset_meta_key = "meta.json"
        self.datamodule_class = "L5MixedDataModule"
        self.dataset_mode = "agents"

        self.rollout.enabled = False
        self.rollout.save_video = True
        self.rollout.every_n_steps = 5000

        # training config
        self.training.batch_size = 100
        self.training.num_steps = 100000
        self.training.num_data_workers = 8

        self.save.every_n_steps = 1000
        self.save.best_k = 10

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32
        self.validation.num_data_workers = 6
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 50


class L5KitMixedEnvConfig(EnvConfig):
    """Vectorized Scene Component + Rasterized Map"""

    def __init__(self):
        super(L5KitMixedEnvConfig, self).__init__()
        self.name = "l5kit"
        # the keys are relative to the dataset environment variable
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"
        self.rasterizer.dataset_meta_key = "meta.json"

        # e.g. 0.0 include every obstacle, 0.5 show those obstacles with >0.5 probability of being
        # one of the classes we care about (cars, bikes, peds, etc.), >=1.0 filter all other agents.
        self.rasterizer.filter_agents_threshold = 0.5

        # whether to completely disable traffic light faces in the semantic rasterizer
        # this disable option is not supported in avsw_semantic
        self.rasterizer.disable_traffic_light_faces = False

        self.generate_agent_obs = False

        self.data_generation_params.other_agents_num = 20
        self.data_generation_params.max_agents_distance = 50
        self.data_generation_params.lane_params.max_num_lanes = 15
        self.data_generation_params.lane_params.max_points_per_lane = 5
        self.data_generation_params.lane_params.max_points_per_crosswalk = 5
        self.data_generation_params.lane_params.max_retrieval_distance_m = 35
        self.data_generation_params.lane_params.max_num_crosswalks = 20
        self.data_generation_params.rasterize_agents = False
        self.data_generation_params.vectorize_agents = True

        # step size of lane interpolation
        self.data_generation_params.lane_params.lane_interp_step_size = 5.0
        self.data_generation_params.vectorize_lane = True

        self.rasterizer.raster_size = (224, 224)

        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = (0.5, 0.5)

        # From 0 to 1 per axis, [0.5,0.5] would show the ego centered in the image.
        self.rasterizer.ego_center = (0.25, 0.5)

        self.rasterizer.map_type = "py_semantic"
        # self.rasterizer.map_type = "scene_semantic"

        # the keys are relative to the dataset environment variable
        self.rasterizer.satellite_map_key = "aerial_map/aerial_map.png"
        self.rasterizer.semantic_map_key = "semantic_map/semantic_map.pb"

        # When set to True, the rasterizer will set the raster origin at bottom left,
        # i.e. vehicles are driving on the right side of the road.
        # With this change, the vertical flipping on the raster used in the visualization code is no longer needed.
        # Set it to False for models trained before v1.1.0-25-g3c517f0 (December 2020).
        # In that case visualisation will be flipped (we've removed the flip there) but the model's input will be correct.
        self.rasterizer.set_origin_to_bottom = True

        #  if a tracked agent is closed than this value to ego, it will be controlled
        self.simulation.distance_th_far = 50

        #  if a new agent is closer than this value to ego, it will be controlled
        self.simulation.distance_th_close = 50

        #  whether to disable agents that are not returned at start_frame_index
        self.simulation.disable_new_agents = False

        # maximum number of simulation steps to run (0.1sec / step)
        self.simulation.num_simulation_steps = 50

        # which frame to start an simulation episode with
        self.simulation.start_frame_index = 0


class L5KitMixedSemanticMapEnvConfig(L5KitMixedEnvConfig):
    def __init__(self):
        super(L5KitMixedSemanticMapEnvConfig, self).__init__()
        self.rasterizer.map_type = "py_semantic"
        self.data_generation_params.vectorize_lane = False
        self.generate_agent_obs = True

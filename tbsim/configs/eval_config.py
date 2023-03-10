import numpy as np
from copy import deepcopy

from tbsim.configs.config import Dict


class EvaluationConfig(Dict):
    def __init__(self):
        super(EvaluationConfig, self).__init__()
        self.name = None
        self.env = "nusc"  # [l5kit, nusc]
        self.dataset_path = None
        self.eval_class = ""
        self.seed = 0
        self.num_scenes_per_batch = 2
        self.num_scenes_to_evaluate = 2

        self.num_episode_repeats = 1
        self.start_frame_index_each_episode = None  # if specified, should be the same length as num_episode_repeats
        self.seed_each_episode = None  # if specified, should be the same length as num_episode_repeats

        self.ego_only = False
        self.agent_eval_class = None

        self.ckpt_root_dir = "checkpoints/"
        self.experience_hdf5_path = None
        self.results_dir = "results/"


        self.ckpt.policy.ckpt_dir = None
        self.ckpt.policy.ckpt_key = None

        self.ckpt.planner.ckpt_dir = None
        self.ckpt.planner.ckpt_key = None

        self.ckpt.predictor.ckpt_dir = None
        self.ckpt.predictor.ckpt_key = None

        self.ckpt.cvae_metric.ckpt_dir = None
        self.ckpt.cvae_metric.ckpt_key = None

        self.ckpt.occupancy_metric.ckpt_dir = None
        self.ckpt.occupancy_metric.ckpt_key = None

        self.policy.mask_drivable = True
        self.policy.num_plan_samples = 50
        self.policy.num_action_samples = 10
        self.policy.pos_to_yaw = True
        self.policy.yaw_correction_speed = 1.0
        self.policy.diversification_clearance = None
        self.policy.sample = True


        self.policy.cost_weights.collision_weight = 15.0
        self.policy.cost_weights.lane_weight = 1.0
        self.policy.cost_weights.lane_dir_weight = 1.0
        self.policy.cost_weights.likelihood_weight = 0.0  # 0.1
        self.policy.cost_weights.progress_weight = 0.01  # 0.005

        self.metrics.compute_analytical_metrics = True
        self.metrics.compute_learned_metrics = False

        self.perturb.enabled = False
        self.perturb.OU.theta = 0.8
        self.perturb.OU.sigma = [0.0, 0.1,0.2,0.5,1.0,2.0,4.0]
        self.perturb.OU.scale = [1.0,1.0,0.2]

        self.rolling_perturb.enabled = False
        self.rolling_perturb.OU.theta = 0.8
        self.rolling_perturb.OU.sigma = 0.5
        self.rolling_perturb.OU.scale = [1.0,1.0,0.2]

        self.occupancy.rolling = True
        self.occupancy.rolling_horizon = [5,10,20]

        self.cvae.rolling = True
        self.cvae.rolling_horizon = [5,10,20]

        self.nusc.eval_scenes = np.arange(0,100).tolist()
        self.nusc.n_step_action = 2
        self.nusc.num_simulation_steps = 10
        self.nusc.skip_first_n = 0
        
        self.drivesim.eval_scenes = np.arange(0,40).tolist()
        self.drivesim.n_step_action = 3
        self.drivesim.num_simulation_steps = 200
        self.drivesim.skip_first_n = 9

        self.l5kit.eval_scenes = [9058, 5232, 14153, 8173, 10314, 7027, 9812, 1090, 9453, 978, 10263, 874, 5563, 9613, 261, 2826, 2175, 9977, 6423, 1069, 1836, 8198, 5034, 6016, 2525, 927, 3634, 11806, 4911, 6192, 11641, 461, 142, 15493, 4919, 8494, 14572, 2402, 308, 1952, 13287, 15614, 6529, 12, 11543, 4558, 489, 6876, 15279, 6095, 5877, 8928, 10599, 16150, 11296, 9382, 13352, 1794, 16122, 12429, 15321, 8614, 12447, 4502, 13235, 2919, 15893, 12960, 7043, 9278, 952, 4699, 768, 13146, 8827, 16212, 10777, 15885, 11319, 9417, 14092, 14873, 6740, 11847, 15331, 15639, 11361, 14784, 13448, 10124, 4872, 3567, 5543, 2214, 7624, 10193, 7297, 1308, 3951, 14001]
        self.l5kit.n_step_action = 5
        self.l5kit.num_simulation_steps = 100
        self.l5kit.skip_first_n = 1
        self.l5kit.skimp_rollout = False

        self.adjustment.enabled = False
        self.adjustment.random_init_plan=False
        self.adjustment.remove_existing_neighbors = False
        self.adjustment.initial_num_neighbors = 4
        self.adjustment.num_frame_per_new_agent = 2000

    def clone(self):
        return deepcopy(self)


class TrainTimeEvaluationConfig(EvaluationConfig):
    def __init__(self):
        super(TrainTimeEvaluationConfig, self).__init__()

        self.num_scenes_per_batch = 4
        self.nusc.eval_scenes = np.arange(0, 100, 10).tolist()
        self.l5kit.eval_scenes = self.l5kit.eval_scenes[:20]

        self.policy.sample = False


import argparse
from copy import deepcopy

from collections import OrderedDict
import os
import torch
from torch.utils.data import DataLoader

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer

from tbsim.l5kit.vectorizer import build_vectorizer
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.algos.algos import BehaviorCloning, VAETrafficModel, BehaviorCloningGC, SpatialPlanner
from tbsim.algos.multiagent_algos import MATrafficModel
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.envs.env_l5kit import EnvL5KitSimulation, BatchedEnv
from tbsim.utils.config_utils import translate_l5kit_cfg, get_experiment_config_from_file
from tbsim.utils.env_utils import rollout_episodes
from tbsim.policies.wrappers import PolicyWrapper, SamplingPolicyWrapper, HierarchicalSamplerWrapper, RolloutWrapper

from tbsim.utils.tensor_utils import to_torch, to_numpy
from tbsim.l5kit.l5_ego_dataset import EgoDatasetMixed, EgoReplayBufferMixed, ExperienceIterableWrapper
from tbsim.utils.experiment_utils import get_checkpoint
from tbsim.utils.vis_utils import build_visualization_rasterizer_l5kit
from imageio import get_writer
from tbsim.utils.timer import Timers

import tbsim.utils.planning_utils as PlanUtils


def run_checkpoint(ckpt_dir="checkpoints/", video_dir="videos/"):
    # policy_ckpt_path, policy_config_path = get_checkpoint(
    #     # ngc_job_id="2646092",  # gcvae_dynUnicycle_yrl0.1_gcTrue_vaeld4_klw0.001_rlFalse
    #     # ckpt_key="iter72999_",
    #     # ckpt_key="iter120999",
    #     # ngc_job_id="2596419",  # gc_clip_regyaw_dynUnicycle_decmlp128,128_decstateTrue_yrl1.0
    #     ngc_job_id="2717287",
    #     ckpt_key="iter8999",
    #     ckpt_root_dir=ckpt_dir
    # )
    # policy_cfg = get_experiment_config_from_file(policy_config_path)

    planner_ckpt_path, planner_config_path = get_checkpoint(
        ngc_job_id="2573128",  # spatial_archresnet50_bs64_pcl1.0_pbl0.0_rlFalse
        ckpt_key="iter55999_",
        ckpt_root_dir=ckpt_dir
    )
    planner_cfg = get_experiment_config_from_file(planner_config_path)

    predictor_ckpt_path, predictor_config_path = get_checkpoint(
        ngc_job_id="2732861",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
        ckpt_key="iter20999",
        ckpt_root_dir=ckpt_dir
    )
    # predictor_ckpt_path, predictor_config_path = get_checkpoint(
    #     ngc_job_id="2717287",  # aaplan_dynUnicycle_yrl0.1_roiFalse_gcTrue_rlayerlayer2_rlFalse
    #     ckpt_key="iter8999",
    #     ckpt_root_dir=ckpt_dir
    # )
    predictor_cfg = get_experiment_config_from_file(predictor_config_path)

    # print(policy_ckpt_path)
    # print(policy_config_path)
    print(planner_ckpt_path)
    print(planner_config_path)
    print(predictor_ckpt_path)
    print(predictor_config_path)

    data_cfg = get_experiment_config_from_file(predictor_config_path)
    assert data_cfg.env.rasterizer.map_type == "py_semantic"
    # os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath("/home/yuxiaoc/repos/l5kit/prediction-dataset")
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath("/home/yuxiaoc/repos/l5kit/prediction-dataset")
    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(data_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)
    eval_zarr = ChunkedDataset(dm.require(
        data_cfg.train.dataset_valid_key)).open()
    env_dataset = EgoDatasetMixed(l5_config, eval_zarr, vectorizer, rasterizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modality_shapes = OrderedDict(
        image=[rasterizer.num_channels()] + data_cfg.env.rasterizer.raster_size)

    planner = SpatialPlanner.load_from_checkpoint(
        planner_ckpt_path,
        algo_config=planner_cfg.algo,
        modality_shapes=modality_shapes,
    ).to(device).eval()

    predictor = MATrafficModel.load_from_checkpoint(
        predictor_ckpt_path,
        algo_config=predictor_cfg.algo,
        modality_shapes=modality_shapes
    ).to(device).eval()

    # if False:
    #     # Option 1: Deterministic planner -> goal-conditional VAE action sampler
    #     controller = L5VAETrafficModel.load_from_checkpoint(
    #         policy_ckpt_path,
    #         algo_config=policy_cfg.algo,
    #         modality_shapes=modality_shapes
    #     ).to(device).eval()
    #
    #     sampler = PolicyWrapper.wrap_controller(
    #         controller, sample=True, num_action_samples=10)
    #     planner = PolicyWrapper.wrap_planner(
    #         planner, mask_drivable=True, sample=False)
    #
    #     sampler = HierarchicalPolicy(planner, sampler)
    #     policy = SamplingPolicy(
    #         ego_action_sampler=sampler, agent_traj_predictor=predictor)
    #
    # if False:
    #     # Option 2: Stochastic planner -> goal-conditional controller
    #     controller = L5TrafficModelGC.load_from_checkpoint(
    #         policy_ckpt_path,
    #         algo_config=policy_cfg.algo,
    #         modality_shapes=modality_shapes
    #     ).to(device).eval()
    #     plan_sampler = PolicyWrapper.wrap_planner(
    #         planner, mask_drivable=True, sample=True, num_plan_samples=10)
    #     sampler = HierarchicalSampler(plan_sampler, controller)
    #     policy = SamplingPolicy(
    #         ego_action_sampler=sampler, agent_traj_predictor=predictor)
    if True:
        # Option 3: Stochastic planner -> agentaware controller

        # controller = MATrafficModel.load_from_checkpoint(
        #     policy_ckpt_path,
        #     algo_config=policy_cfg.algo,
        #     modality_shapes=modality_shapes
        # ).to(device).eval()
        controller = predictor
        plan_sampler = PolicyWrapper.wrap_planner(
            planner, mask_drivable=True, sample=True, num_plan_samples=10)
        sampler = HierarchicalSamplerWrapper(plan_sampler, controller)
        policy = SamplingPolicyWrapper(
            ego_action_sampler=sampler, agent_traj_predictor=predictor)

    policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)
    # policy = RolloutWrapper(ego_policy=policy)

    dm = LocalDataManager(None)
    l5_config = deepcopy(l5_config)
    l5_config["raster_params"]["raster_size"] = (1000, 1000)
    l5_config["raster_params"]["pixel_size"] = (0.1, 0.1)
    render_rasterizer = build_visualization_rasterizer_l5kit(l5_config, dm)
    data_cfg.env.simulation.num_simulation_steps = 200
    data_cfg.env.simulation.distance_th_far = 1e+5
    data_cfg.env.simulation.disable_new_agents = True
    data_cfg.env.generate_agent_obs = True

    num_scenes_per_batch = 1
    env = EnvL5KitSimulation(
        data_cfg.env,
        dataset=env_dataset,
        seed=4,
        num_scenes=num_scenes_per_batch,
        prediction_only=False,
        renderer=render_rasterizer,
        compute_metrics=True,
        skimp_rollout=False
    )

    stats, info, renderings, _,_ = rollout_episodes(
        env,
        policy,
        num_episodes=1,
        n_step_action=3,
        render=True,
        skip_first_n=1,
        # scene_indices=[11, 16, 35, 38, 45, 58, 150, 152, 154, 156],
        #scene_indices=[35, 45, 58],
        scene_indices=[10206],
        # scene_indices=[2772, 10206, 13734, 14248, 15083, 15147, 15453]
        # scene_indices=[150, 1652, 2258, 3496, 14962, 15756]
    )
    print(stats)

    for i, scene_images in enumerate(renderings[0]):
        writer = get_writer(os.path.join(
            video_dir, "{}.mp4".format(info["scene_index"][i])), fps=10)
        for im in scene_images:
            writer.append_data(im)
        writer.close()



def test_sample_planner():
    os.environ["L5KIT_DATA_FOLDER"] = "/home/yuxiaoc/repos/l5kit/prediction-dataset"
    config_file = "/home/yuxiaoc/repos/behavior-generation/experiments/templates/l5_ma_rasterized_plan.json"
    pred_cfg = get_experiment_config_from_file(config_file)

    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = os.path.abspath(
        pred_cfg.train.dataset_path)
    dm = LocalDataManager(None)
    l5_config = translate_l5kit_cfg(pred_cfg)
    rasterizer = build_rasterizer(l5_config, dm)
    vectorizer = build_vectorizer(l5_config, dm)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_zarr = ChunkedDataset(dm.require(
        pred_cfg.train.dataset_valid_key)).open()
    train_dataset = EgoDatasetMixed(
        l5_config, train_zarr, vectorizer, rasterizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=2,
        num_workers=1,
    )
    tr_it = iter(train_dataloader)
    batch = next(tr_it)

    for key, obj in batch.items():
        batch[key] = obj.to(device)

    # model(batch)
    N = 10
    ego_trajectories = torch.cat(
        (batch["target_positions"], batch["target_yaws"]), -1)
    ego_trajectories = ego_trajectories.unsqueeze(1).repeat(1, N, 1, 1)
    ego_trajectories += torch.normal(
        torch.zeros_like(ego_trajectories), torch.ones_like(
            ego_trajectories) * 0.5
    )
    agent_trajectories = torch.cat(
        (
            batch["all_other_agents_future_positions"],
            batch["all_other_agents_future_yaws"],
        ),
        -1,
    )
    raw_types = batch["all_other_agents_types"]
    agent_extents = batch["all_other_agents_future_extents"][..., :2].max(
        dim=-2)[0]
    lane_mask = (batch["image"][:, -3] < 1.0).type(torch.float)
    dis_map = GeoUtils.calc_distance_map(lane_mask)

    col_loss = PlanUtils.get_collision_loss(
        ego_trajectories,
        agent_trajectories,
        batch["extent"][:, :2],
        agent_extents,
        raw_types,
    )

    lane_loss = PlanUtils.get_drivable_area_loss(
        ego_trajectories,
        batch["centroid"],
        batch["yaw"],
        batch["raster_from_world"],
        dis_map,
        batch["extent"][:, :2],
    )
    idx = PlanUtils.ego_sample_planning(
        ego_trajectories,
        agent_trajectories,
        batch["extent"][:, :2],
        agent_extents,
        raw_types,
        batch["raster_from_agent"],
        dis_map,
        weights={"collision_weight": 1.0, "lane_weight": 1.0},
    )


if __name__ == "__main__":
    run_checkpoint()
    # test_sample_planner()

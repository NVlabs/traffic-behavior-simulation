"""A script for evaluating closed-loop simulation"""
import argparse
import numpy as np
import json
import random
import yaml
import importlib
from collections import Counter
from pprint import pprint

import os
from tbsim.utils.metrics import OrnsteinUhlenbeckPerturbation
import torch

from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.configs.eval_config import EvaluationConfig
from tbsim.utils.env_utils import rollout_episodes
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvL5Builder, EnvDrivesimBuilder

from tbsim.policies.wrappers import (
    RolloutWrapper,
    Pos2YawWrapper,
    PerturbationWrapper
)

from tbsim.utils.tensor_utils import map_ndarray
from imageio import get_writer
import pickle

def run_evaluation(eval_cfg, save_cfg, data_to_disk, render_to_video):
    
    if eval_cfg.env in ["nusc","drivesim"]:
        set_global_batch_type("trajdata")
    elif eval_cfg.env == 'l5kit':
        set_global_batch_type("l5kit")

    print(eval_cfg)

    # for reproducibility
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)

    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)
    os.makedirs(os.path.join(eval_cfg.results_dir, "videos/"), exist_ok=True)
    os.makedirs(eval_cfg.ckpt_root_dir, exist_ok=True)
    if save_cfg:
        json.dump(eval_cfg, open(os.path.join(eval_cfg.results_dir, "config.json"), "w+"))
    if data_to_disk and os.path.exists(eval_cfg.experience_hdf5_path):
        os.remove(eval_cfg.experience_hdf5_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper
    
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
    policy, exp_config = composer.get_policy()

    if eval_cfg.policy.pos_to_yaw:
        policy = Pos2YawWrapper(
            policy,
            dt=exp_config.algo.step_time,
            yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
        )
    if eval_cfg.rolling_perturb.enabled:
        OU_pert = OrnsteinUhlenbeckPerturbation(theta=eval_cfg.rolling_perturb.OU.theta*np.ones(3),
                    sigma=eval_cfg.rolling_perturb.OU.sigma*np.array(eval_cfg.rolling_perturb.OU.scale))
        policy = PerturbationWrapper(policy,OU_pert)
    
    if eval_cfg.agent_eval_class is not None:
        composer_class = getattr(policy_composers, eval_cfg.agent_eval_class)
        composer = composer_class(eval_cfg, device, ckpt_root_dir=eval_cfg.ckpt_root_dir)
        agent_policy, _ = composer.get_policy()
        if eval_cfg.policy.pos_to_yaw:
            agent_policy = Pos2YawWrapper(
                agent_policy,
                dt=exp_config.algo.step_time,
                yaw_correction_speed=eval_cfg.policy.yaw_correction_speed
            )
    else:
        agent_policy = None
    
    if eval_cfg.env in ["nusc","drivesim"]:
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(agents_policy=policy)
    elif eval_cfg.ego_only:
        rollout_policy = RolloutWrapper(ego_policy=policy)
    else:
        if eval_cfg.agent_eval_class is not None:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=agent_policy)
        else:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)
    
    print(exp_config.algo)

    # create env
    if eval_cfg.env == "nusc":
        if agent_policy is not None:
            split_ego = True
        else:
            split_ego = False
        env_builder = EnvNuscBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        if "parse_obs" in exp_config.env.data_generation_params:
            parse_obs=exp_config.env.data_generation_params.parse_obs
        else:
            parse_obs=True
        env = env_builder.get_env(split_ego=split_ego,parse_obs=parse_obs)
    elif eval_cfg.env == "drivesim":
        if agent_policy is not None:
            split_ego = True
        else:
            split_ego = False
        env_builder = EnvDrivesimBuilder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        if "parse_obs" in exp_config.env.data_generation_params:
            parse_obs=exp_config.env.data_generation_params.parse_obs
        else:
            parse_obs=True
        env = env_builder.get_env(split_ego=split_ego,parse_obs=parse_obs)
    elif eval_cfg.env == 'l5kit':
        env_builder = EnvL5Builder(eval_config=eval_cfg, exp_config=exp_config, device=device)
        env = env_builder.get_env()
    else:
        raise NotImplementedError("{} is not a valid env".format(eval_cfg.env))

    

    # eval loop
    obs_to_torch = eval_cfg.eval_class not in ["GroundTruth", "ReplayAction"]

    result_stats = None
    scene_i = 0
    eval_scenes = eval_cfg.eval_scenes
    total_adjust_plan = dict()
    total_trace = dict()
    total_info = dict()
    while scene_i < eval_cfg.num_scenes_to_evaluate:
        scene_indices = eval_scenes[scene_i: scene_i + eval_cfg.num_scenes_per_batch]
        scene_i += eval_cfg.num_scenes_per_batch

        stats, info, renderings, adjust_plans, trace = rollout_episodes(
            env,
            rollout_policy,
            num_episodes=eval_cfg.num_episode_repeats,
            n_step_action=eval_cfg.n_step_action,
            render=render_to_video,
            skip_first_n=eval_cfg.skip_first_n,
            scene_indices=scene_indices,
            obs_to_torch=obs_to_torch,
            start_frame_index_each_episode=eval_cfg.start_frame_index_each_episode,
            seed_each_episode=eval_cfg.seed_each_episode,
            horizon=eval_cfg.num_simulation_steps,
            adjust_plan_recipe=eval_cfg.adjustment.to_dict() if eval_cfg.adjustment.enabled else None,
        )
        for ei,adjust_plan in enumerate(adjust_plans):
            for k,v in adjust_plan.items():
                total_adjust_plan["{}_{}".format(k,ei)]=v 
        for ei,trace_i in enumerate(trace):
            for k,v in trace_i.items():
                total_trace["{}_{}".format(k,ei)]=v

    

        print(info["scene_index"])
        pprint(stats)

        # aggregate metrics stats
        if result_stats is None:
            result_stats = stats
            result_stats["scene_index"] = np.array(info["scene_index"])
        else:
            for k in stats:
                result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
            result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

        # write stats to disk
        with open(os.path.join(eval_cfg.results_dir, "stats.json"), "w+") as fp:
            stats_to_write = map_ndarray(result_stats, lambda x: x.tolist())
            json.dump(stats_to_write, fp)

        if render_to_video:
            for ei, episode_rendering in enumerate(renderings):
                for i, scene_images in enumerate(episode_rendering):
                    video_dir = os.path.join(eval_cfg.results_dir, "videos/")
                    writer = get_writer(os.path.join(
                        video_dir, "{}_{}.mp4".format(info["scene_index"][i], ei)), fps=10)
                    print("video to {}".format(os.path.join(
                        video_dir, "{}_{}.mp4".format(info["scene_index"][i], ei))))
                    for im in scene_images:
                        writer.append_data(im)
                    writer.close()

        if data_to_disk and "buffer" in info:
            dump_episode_buffer(
                info["buffer"],
                info["scene_index"],
                h5_path=eval_cfg.experience_hdf5_path
            )
        torch.cuda.empty_cache()
        if len(total_adjust_plan)>0:
            with open(os.path.join(eval_cfg.results_dir, "adjust_plan.json"),"w+") as fp:
                json.dump(total_adjust_plan,fp)
                print("adjust plan saved to {}".format(os.path.join(eval_cfg.results_dir, "adjust_plan.json")))
        
        info_except_buffer = {k:v for k,v in info.items() if k!="buffer"}
        for k,v in info_except_buffer.items():
            if k not in total_info:
                total_info[k]=v
            else:
                if isinstance(v,list):
                    total_info[k].extend(v)
                elif isinstance(v,dict):
                    total_info[k].update(v)
        try:
            with open(os.path.join(eval_cfg.results_dir, "sim_info.json"),"w+") as fp:
                json.dump(total_info,fp)
                print("sim info saved to {}".format(os.path.join(eval_cfg.results_dir, "sim_info.json")))
        except:
            print("fail to save sim info")
        if len(total_trace)>0:
            with open(os.path.join(eval_cfg.results_dir, "trace.pkl"),"wb") as fp:
                pickle.dump(total_trace,fp)
                print("trace saved to {}".format(os.path.join(eval_cfg.results_dir, "trace.pkl")))


def dump_episode_buffer(buffer, scene_index, h5_path):
    import h5py
    h5_file = h5py.File(h5_path, "a")

    ep_count = Counter()
    for si, scene_buffer in zip(scene_index, buffer):
        # TODO: fix this hack
        # Postfix scene index with episode count (scene may repeat with multiple episodes)
        ep_i = ep_count[si]
        ep_count[si] += 1
        for mk in scene_buffer:
            h5key = "/{}_{}/{}".format(si, ep_i, mk)
            h5_file.create_dataset(h5key, data=scene_buffer[mk])
    h5_file.close()
    print("scene {} written to {}".format(scene_index, h5_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank for torch.distributed"
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "l5kit", "drivesim"],
        help="Which env to run evaluation in",
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None
    )

    parser.add_argument(
        "--metric_ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location for the learned metric",
        default=None
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--agent_eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class for agents if it's different from ego"
    )

    parser.add_argument(
        "--ckpt_root_dir",
        type=str,
        default=None,
        help="Root directory to look for training run directories"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=None,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="whether to render videos"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    cfg = EvaluationConfig()

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key

    if args.num_scenes_per_batch is not None:
        cfg.num_scenes_per_batch = args.num_scenes_per_batch

    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class

    if args.prefix is not None:
        cfg.name = args.prefix + cfg.name

    if args.agent_eval_class is not None:
        cfg.agent_eval_class = args.agent_eval_class

    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    if args.env is not None:
        cfg.env = args.env
    else:
        assert cfg.env is not None

    cfg.experience_hdf5_path = os.path.join(cfg.results_dir, "data.hdf5")

    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("drivesim")
    cfg.pop("l5kit")

    if args.ckpt_yaml is not None:
        with open(args.ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    if args.metric_ckpt_yaml is not None:
        with open(args.metric_ckpt_yaml, "r") as f:
            ckpt_info = yaml.safe_load(f)
            cfg.ckpt.update(**ckpt_info)
    
    cfg.lock()
    run_evaluation(
        cfg,
        save_cfg=True,
        data_to_disk=True,
        render_to_video=args.render
    )

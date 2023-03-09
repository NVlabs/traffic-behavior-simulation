"""A script for launching training runs on NGC"""
import argparse
import os.path

import yaml
from tbsim.configs.eval_config import EvaluationConfig

from tbsim.utils.experiment_utils import ParamConfig, create_evaluation_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
    if base_cfg.env=="l5kit":
        if base_cfg.eval_class not in ["HierAgentAwareMPC", "HAASplineSampling"]:
            plan.extend(plan.compose_cartesian([
                                                ParamRange("l5kit.num_simulation_steps", alias="horizon", range=[200]),
                                                ParamRange("num_episode_repeats", alias="repeats", range=[4]),
                                                ParamRange("rolling_perturb.enabled", alias="rp", range=[False]),
                                                ParamRange("policy.pos_to_yaw", alias="p2y", range=[True]),
                                                ParamRange("policy.cost_weights.collision_weight", alias="coll_weight", range=[50.0,20.0,10.0,3.0]),
                                                # ParamRange("rolling_perturb.OU.sigma", alias="sigma", range=[0.0,0.1,0.2,0.5,1.0,2.0]),
                                                ]))
        else:
            plan.extend(plan.compose_cartesian([
                                                ParamRange("l5kit.num_simulation_steps", alias="horizon", range=[200]),
                                                ParamRange("perturb.enabled", alias="p", range=[False]),
                                                ParamRange("num_episode_repeats", alias="repeats", range=[4]),
                                                ParamRange("cvae.rolling", alias="cr", range=[True]),
                                                ParamRange("occupancy.rolling", alias="or", range=[True]),
                                                ParamRange("rolling_perturb.enabled", alias="rp", range=[False]),
                                                ParamRange("policy.pos_to_yaw", alias="p2y", range=[False]),
                                                ParamRange("agent_eval_class", alias="ac", range=["HierAgentAware","BC","TrafficSim","TPP"]),
                                                ]))
        
    elif base_cfg.env=="nusc":
        plan.extend(plan.compose_cartesian([
                                            ParamRange("nusc.num_simulation_steps", alias="horizon", range=[200]),
                                            ParamRange("perturb.enabled", alias="p", range=[False]),
                                            ParamRange("num_episode_repeats", alias="repeats", range=[4]),
                                            ParamRange("cvae.rolling", alias="cr", range=[True]),
                                            ParamRange("occupancy.rolling", alias="or", range=[True]),
                                            ParamRange("rolling_perturb.enabled", alias="rp", range=[True]),
                                            ParamRange("policy.pos_to_yaw", alias="p2y", range=[True]),
                                            ParamRange("rolling_perturb.OU.sigma", alias="sigma", range=[0.0,0.1,0.2,0.5,1.0,2.0]),
                                            ]))
    return plan.generate_configs(base_cfg=base_cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used as the template for parameter tuning"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="experiments/test/",
        help="directory for saving generated config files."
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix of the experiment names"
    )

    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        required=True
    )

    parser.add_argument(
        "--ckpt_yaml",
        type=str,
        help="specify a yaml file that specifies checkpoint and config location of each model",
        default=None,
        required=True
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=["nusc", "l5kit"],
        help="Which env to run evaluation in",
        required=True
    )

    parser.add_argument(
        "--ckpt_root_dir",
        type=str,
        default=None,
        help="root directory of checkpoints",
        required=False
    )

    args = parser.parse_args()
    cfg = EvaluationConfig()
    name = os.path.basename(args.ckpt_yaml)[:-5]
    if args.prefix is not None:
        name = args.prefix + name
    cfg.name = name

    cfg.eval_class = args.eval_class
    cfg.env = args.env
    if args.ckpt_root_dir is not None:
        cfg.ckpt_root_dir = args.ckpt_root_dir
    with open(args.ckpt_yaml, "r") as f:
        ckpt_info = yaml.safe_load(f)
        cfg.ckpt.update(**ckpt_info)

    create_evaluation_configs(
        configs_to_search,
        args.config_dir,
        cfg,
        args.prefix,
        delete_config_dir=False
    )

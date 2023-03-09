"""A script for launching training runs on NGC"""
import argparse
import os.path

import yaml
from tbsim.configs.eval_config import EvaluationConfig

from tbsim.utils.experiment_utils import ParamConfig, create_evaluation_configs, ParamSearchPlan, ParamRange, Param


def configs_to_search(base_cfg):
    """Override this with your hyperparameter search plan"""
    plan = ParamSearchPlan()
        
    assert base_cfg.env=="nusc"
    # plan.extend(plan.compose_zip([
    #                             ParamRange("ckpt.predictor.ngc_job_id", alias="ckpt_id", range=[4130553,4130552]),
    #                             ParamRange("ckpt.predictor.ckpt_key", alias="ckpt_key", range=["94000","94000"]),
    #                             ]))
    plan.add_const_param(Param("ckpt.predictor.ngc_job_id", alias="ckpt_id", value=4130553))
    plan.add_const_param(Param("ckpt.predictor.ckpt_key", alias="ckpt_key", value="94000"))
    plan.extend(plan.compose_cartesian([
                                ParamRange("policy.cost_weights.collision_weight", alias="cw", range=[10.0,5.0,20.0]),
                                ParamRange("policy.cost_weights.progress_weight", alias="pw", range=[0,0.005]),
                                ParamRange("policy.cost_weights.likelihood_weight", alias="lw", range=[0,0.1]),
                                ParamRange("nusc.n_step_action", alias="nstep", range=[2,4]),

    ]))
    plan.extend(plan.compose_zip([
                                ParamRange("policy.num_action_samples", alias="num_sample", range=[10,20]),
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

import argparse
import json
import os

from tbsim.utils.experiment_utils import (
    launch_experiments_ngc,
    upload_codebase_to_ngc_workspace,
    read_evaluation_configs
)
from tbsim.configs.eval_config import EvaluationConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json to launch the experiment with"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="experiments/",
        help="directory to read config files from."
    )

    parser.add_argument(
        "--ngc_config",
        type=str,
        help="path to your ngc config file",
        default="ngc/ngc_config.json"
    )

    parser.add_argument(
        "--render",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "--num_scenes_per_batch",
        type=int,
        default=4,
        help="Number of scenes to run concurrently (to accelerate eval)"
    )

    parser.add_argument(
        "--ckpt_root_dir",
        type=str,
        default=None,
        help="checkpoint root directory when running on ngc"
    )

    parser.add_argument(
        "--script_path",
        type=str,
        default="scripts/evaluate.py"
    )

    parser.add_argument(
        "--ngc_instance",
        type=str,
        default="dgx1v.16g.1.norm"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False
    )

    args = parser.parse_args()
    if args.config_file is not None:
        cfg = EvaluationConfig()
        ext_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**ext_cfg)
        cfgs = [cfg]
        cfg_fns = [args.config_file]
    else:
        cfgs, cfg_fns = read_evaluation_configs(args.config_dir)

    ngc_cfg = json.load(open(args.ngc_config, "r"))
    ngc_cfg["instance"] = args.ngc_instance

    script_command = [
        "python",
        args.script_path,
        "--results_root_dir",
        ngc_cfg["eval_output_dir"],
        "--num_scenes_per_batch",
        str(args.num_scenes_per_batch),
        "--dataset_path",
        ngc_cfg["dataset_path"],
        "--env",
        cfgs[0].env
    ]
    if args.ckpt_root_dir is not None:
        script_command+=["--ckpt_root_dir",
        args.ckpt_root_dir]

    if args.render:
        script_command.append("--render")

    res = input("make sure you have synced your code to ngc workspace! (enter to continue)")
    launch_experiments_ngc(script_command, cfgs, cfg_fns, ngc_config=ngc_cfg, dry_run=args.dry_run)
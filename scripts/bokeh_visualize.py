import argparse
import os

def run_bokeh(args):
    result_dir = os.path.join(args.results_root_dir,args.eval_name)
    command = f"bokeh serve --show tbsim/utils/bokeh_script.py --args -result_dir {result_dir} -ei {args.episode}"
    if args.scene_name is not None:
        command +=f" -scene_name {args.scene_name}"
    os.system(command)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--eval_name",
        type=str,
        default=None,
        help="Specify the evaluation class through argparse"
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--results_root_dir",
        type=str,
        default="/home/yuxiaoc/repos/behavior-generation/results",
        help="Directory to save results and videos"
    )


    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="episode to visualize"
    )


    args = parser.parse_args()
    run_bokeh(args)

    
   

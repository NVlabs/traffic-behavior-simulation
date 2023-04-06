# Traffic Behavior Simulation (tbsim)
TBSIM is a simulation environment designed for data-driven closed-loop simulation of autonomous vehicles. It supports training and evaluation of popular traffic models such as behavior cloning, CVAE, and our new [BITS](https://arxiv.org/abs/2208.12403) model specifically designed for AV simulation. The users can flexibly specify the simulation environment and plug in their own model (learned or analytic) for evaluation.

Thanks to [trajdata](https://github.com/NVlabs/trajdata), TBSIM can access data and scenarios from a wide range of public datasets, including [Lyft Level 5](https://woven.toyota/en/prediction-dataset), [nuScenes](https://www.nuscenes.org/nuscenes), and [nuPlan](https://nuplan.org/).

TBSIM is well equiped with abundant util functions, and supports batched simulation in parallel, logging, and replay. We also provide a suite of simulation metrics that measures the safety, liveness, and diversity of the simulation.

<img src="assets/sample_rollout.gif" width="750" height="350"/>

## Installation

Install `tbsim`
```angular2html
conda create -n tbsim python=3.8
conda activate tbsim
git clone git@github.com:NVlabs/traffic-behavior-simulation.git tbsim
cd tbsim
pip install -e .
```

Install `trajdata`
```
cd ..
git clone ssh://git@github.com:NVlabs/trajdata.git trajdata
cd trajdata
# replace requirements.txt with trajdata_requirements.txt included in tbsim
pip install -e .
```

Install `Pplan`
```
cd ..
git clone ssh://git@github.com:NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
```

Usually the user needs to install torch separately that fits the hardware setup (OS, GPU, CUDA version, etc., check https://pytorch.org/get-started/locally/ for instructions)
## Quick start
### 1. Obtain dataset(s)
We currently support the Lyft Level 5 [dataset](https://woven.toyota/en/prediction-dataset) and the nuScenes [dataset](https://www.nuscenes.org/nuscenes).

#### Lyft Level 5:
* Download the Lyft Prediction dataset (only the metadata and the map) and organize the dataset directory as follows:
    ```
    lyft_prediction/
    │   aerial_map/
    │   semantic_map/
    │   meta.json
    └───scenes
    │   │   sample.zarr
    │   │   train_full.zarr
    │   │   train.zarr
    |   |   validate.zarr
    ```

#### nuScenes
* Download the nuScenes dataset (with the v1.3 map extension pack) and organize the dataset directory as follows:
    ```
    nuscenes/
    │   maps/
    │   v1.0-mini/
    │   v1.0-trainval/
    ```
### 2. Train a behavior cloning model
Lyft dataset (set `--debug` flag to suppress wandb logging):
```
python scripts/train.py --dataset_path <path-to-lyft-data-directory> --config_name l5_bc --debug
```

nuScenes dataset (set `--debug` flag to suppress wandb logging):
```
python scripts/train.py --dataset_path <path-to-nuscenes-data-directory> --config_name nusc_bc --debug
```

See the list of registered algorithms in `configs/registry.py`

### 3. Train BITS model

Lyft dataset:

First train a spatial planner:
```
python scripts/train.py --dataset_path <path-to-lyft-data-directory> --config_name l5_spatial_planner --debug
```
Then train a multiagent predictor:
```
python scripts/train.py --dataset_path <path-to-lyft-data-directory> --config_name l5_agent_predictor --debug
```

nuScenes dataset:
First train a spatial planner:
```
python scripts/train.py --dataset_path <path-to-nuScenes-data-directory> --config_name nusc_spatial_planner --debug
```
Then train a multiagent predictor:
```
python scripts/train.py --dataset_path <path-to-nuScenes-data-directory> --config_name nusc_agent_predictor --debug
```

See the list of registered algorithms in `configs/registry.py`
### 4. Evaluate a trained model (closed-loop simulation)
```
python scripts/evaluate.py \
  --results_root_dir results/ \
  --num_scenes_per_batch 2 \
  --dataset_path <your-dataset-path> \
  --env <l5kit|nusc> \
  --policy_ckpt_dir <path-to-checkpoint-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class BC \
  --render
```

### 5. Closed-loop simulation with BITS
With the spatial planner and multiagent predictor trained, one can run BITS simulation with

```
python scripts/evaluate.py \
  --results_root_dir results/ \
  --dataset_path <your-dataset-path> \
  --env <l5kit|nusc> \
  --ckpt_yaml <path-to-yaml-dir> \
  --eval_class HierAgentAware \
  --render
```
The ckpt_yaml file specifies the checkpoints for the spatial planner and predictor, an example can be found at `evaluation/BITS.yaml`.

### 6. Closed-loop evaluation of policy with BITS

TBSIM allows the ego to have a separate policy than the rest of the agents. An example command is

```
python scripts/evaluate.py \
  --results_root_dir results/ \
  --dataset_path <your-dataset-path> \
  --env <l5kit|nusc> \
  --ckpt_yaml <path-to-yaml-dir> \
  --eval_class <your-policy-name> \
  --agent_eval_class=HierAgentAware\
  --render
```

Here your policy should be declared in `tbsim/evaluation/policy_composer.py`.


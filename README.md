# Traffic Behavior Simulation (tbsim)
Software infrastructure for learning-based traffic simulation.

<img src="assets/sample_rollout.gif" width="750" height="250"/>

## Installation

Install `tbsim`
```angular2html
conda env create -n tbsim python=3.9
conda activate tbsim
git clone ssh://git@gitlab-master.nvidia.com:12051/nvr-av/behavior-generation.git tbsim
cd tbsim
pip install -e .
```

Install `trajdata`
```
cd ..
git clone ssh://git@gitlab-master.nvidia.com:12051/nvr-av/unified-av-data-loader.git trajdata
cd trajdata
# replace requirements.txt with trajdata_requirements.txt included in tbsim
pip install -e .
```

Install `Pplan`
```
cd ..
git clone ssh://git@gitlab-master.nvidia.com:12051/nvr-av/Pplan.git Pplan
cd Pplan
pip install -e .
```

## Quick start
### 1. Obtain dataset(s)
We currently support the Lyft Level 5 [dataset](https://level-5.global/data/) and the nuScenes [dataset](https://www.nuscenes.org/nuscenes).

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

### 3. Evaluate a trained model (closed-loop simulation)
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

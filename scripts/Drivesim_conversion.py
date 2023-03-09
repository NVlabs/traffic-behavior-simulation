import h5py
import numpy as np
from pathlib import Path
from trajdata.dataset_specific.drivesim.test import convert_to_DS
import os
from collections import defaultdict


def main():
    selected_scenes = ["scene_0_0","scene_1_0"]
    result_root_dir = Path("results/")
    policy_name = "HierAgentAware"
    track_ids = dict()
    track_ids["scene_0"] =["ego"]+[str(i) for i in range(769,769+17-1)]
    track_ids["scene_1"] =["ego"]+[str(i) for i in range(769,769+17-1)]
    hdf5_path = result_root_dir/policy_name/"data.hdf5"
    if not (result_root_dir/policy_name/"drivesim_files").exists():
        os.mkdir(result_root_dir/policy_name/"drivesim_files")
    h5f = h5py.File(hdf5_path, "r")
    for scene_name,data in h5f.items():
        if selected_scenes is not None and scene_name not in selected_scenes:
            continue
        pos = np.array(data["centroid"])
        yaw = np.array(data["yaw"])[...,None]
        poses = np.concatenate([pos,yaw],-1)
        track_id = data["track_id"][:,0].tolist()
        for k,v in track_ids.items():
            if k in scene_name:
                track_id = v
                break
        out_dict = convert_to_DS(poses, track_id,fps=10)
        np.savez(
        result_root_dir/policy_name/"drivesim_files"/f"{scene_name}.npz",
        **out_dict
    )
        
    
if __name__=="__main__":
    main()
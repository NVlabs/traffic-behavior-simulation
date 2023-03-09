from typing import OrderedDict
import numpy as np
import pytorch_lightning as pl
import torch
import importlib
import os
from imageio import get_writer

from tbsim.envs.base import BatchedEnv, BaseEnv
import tbsim.utils.tensor_utils as TensorUtils
from tbsim.utils.timer import Timers
from tbsim.policies.wrappers import RolloutWrapper
from l5kit.simulation.unroll import ClosedLoopSimulator
import tbsim.utils.geometry_utils as GeoUtils
from tbsim.policies.wrappers import Pos2YawWrapper
from tbsim.evaluation.env_builders import EnvNuscBuilder, EnvL5Builder
from tbsim.utils.trajdata_utils import parse_trajdata_batch
from tbsim.utils.geometry_utils import VEH_VEH_collision
from trajdata.simulation import SimulationScene
import random

def collision_check(agents_posyaw,new_posyaw,agents_extent,new_extent):
    new_posyaw_tiled=new_posyaw[np.newaxis,:].repeat(agents_posyaw.shape[0],0)
    new_extent_tiled=new_extent[np.newaxis,:].repeat(agents_posyaw.shape[0],0)
    dis = VEH_VEH_collision(new_posyaw_tiled,agents_posyaw,new_extent_tiled,agents_extent)
    return dis
def random_placing_neighbors(simscene,num_neighbors,coll_check=True):
    init_modes = [0,1,2,3,4]
    random.shuffle(init_modes)
    init_modes = init_modes[:num_neighbors]
    offset_x = 18.0
    offset_y = 5.0
    T = 10
    v_sigma=0.3

    dt = simscene.scene.dt
    obs = simscene.get_obs()
    if isinstance(simscene,SimulationScene):
        obs = parse_trajdata_batch(obs)

    obs = TensorUtils.to_numpy(obs,ignore_if_unspecified=True)
    num_new_agent = 0
    agent_names = [agent.name for agent in simscene.agents]
    while "agent"+str(num_new_agent) in agent_names:
        num_new_agent+=1
    ego_vel = obs["curr_speed"][0]
    agent_plan = list()
    for i in range(num_neighbors):
        newagent_name="agent"+str(num_new_agent+i)
        newagent_type = 1
        newagent_extent=np.array([4.,2.5,2.])
        if init_modes[i] == 0:
            # in front of the ego vehicle
            newagent_state = np.array([[offset_x,0,0]]).repeat(T,0)
            vel = np.clip(ego_vel-2.0+np.random.randn()*v_sigma,0.,40.)
        elif init_modes[i] == 1:
            # behind of the ego vehicle
            newagent_state = np.array([[-offset_x,0,0]]).repeat(T,0)
            vel = np.clip(ego_vel+2.0+np.random.randn()*v_sigma,0.,40.)
        elif init_modes[i] == 2:
            # left of the ego vehicle
            newagent_state = np.array([[0,-offset_y,0]]).repeat(T,0)
            vel = np.clip(ego_vel+np.random.randn()*v_sigma,0.,40.)
        elif init_modes[i] == 3:
            # right of the ego vehicle
            newagent_state = np.array([[0,offset_y,0]]).repeat(T,0)
            vel = np.clip(ego_vel+np.random.randn()*v_sigma,0.,40.)
        elif init_modes[i] == 4:
            # two vehicle length ahead of the ego vehicle
            newagent_state = np.array([[2*offset_x,0,0]]).repeat(T,0)
            vel = np.clip(ego_vel-4.0+np.random.randn()*v_sigma,0.,40.)
        
        newagent_state[:,0]+=np.arange(-T+1,1)*dt*vel

        add_flag = True
        new_pos_global = GeoUtils.batch_nd_transform_points_np(newagent_state[:,:2],obs["world_from_agent"][0])
        new_yaw_global = newagent_state[:,2:]+obs["yaw"][0]
        newagent_state_global = np.hstack((new_pos_global,new_yaw_global))
        if coll_check:
            
            if "centroid" in obs:
                agents_pos_global = obs["centroid"]
            else:
                agents_pos_global = GeoUtils.batch_nd_transform_points_np(obs["history_positions"][:,-1],obs["world_from_agent"])
            agents_yaw_global = obs["yaw"]
            agents_posyaw = np.hstack((agents_pos_global,agents_yaw_global[:,np.newaxis]))
            new_posyaw = np.hstack((new_pos_global[0],new_yaw_global[0]))

            dis = collision_check(agents_posyaw,new_posyaw,obs["extent"],newagent_extent)
            if dis.min()<2.0:
                add_flag = False

        if add_flag:
            agent_plan.append(dict(name=newagent_name,
                                agent_state=newagent_state_global.tolist(),
                                initial_timestep=simscene.scene_ts-T+1,
                                agent_type=newagent_type,
                                extent=newagent_extent.tolist(),
                                executed=False))
    return agent_plan

def random_initial_adjust_plan(env,adjust_recipe):
    adjust_plan = dict()
    for simscene in env._current_scenes:
        adjust_plan[simscene.scene.name] = dict(remove_existing_neighbors=dict(flag=adjust_recipe["remove_existing_neighbors"],executed=False),
                                              agents=random_placing_neighbors(simscene,adjust_recipe["initial_num_neighbors"]))

    return adjust_plan

def set_initial_states(env, obs, adjust_plan, device):
    """A function that sets initial states of an env based on an observation dictionary (TODO: clean up)"""
    obs = TensorUtils.to_torch(obs, device)
    bs, T = obs["ego"]["target_positions"].shape[:2]
    
    ego_global = GeoUtils.batch_nd_transform_points(
        obs["ego"]["history_positions"][:, 0],
        obs["ego"]["world_from_agent"]
    )
    ego_yaw = obs["ego"]["history_yaws"][:, 0].flatten()+obs["ego"]["yaw"]
    offset_x = 8.0
    offset_y = 4.0
    agent_indices = []
    positions = []
    yaws = []
    for i in range(bs):
        scene_idx = obs["ego"]["scene_index"][i].item()
        if scene_idx in adjust_plan:
            for agent_id in adjust_plan[scene_idx]:
                agent_idx = torch.where(
                    (obs["agents"]["scene_index"] == scene_idx) &
                    (obs["agents"]["track_id"] == agent_id)
                )[0]
                if agent_idx.nelement()==1:
                    agent_idx = agent_idx.item()
                else:
                    continue
                agent_indices.append(agent_idx)
                
                if adjust_plan[scene_idx][agent_id] == 0:
                    # in front of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [offset_x*torch.cos(ego_yaw[i]), offset_x*torch.sin(ego_yaw[i])]).to(device)
                elif adjust_plan[scene_idx][agent_id] == 1:
                    # behind of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [-offset_x*torch.cos(ego_yaw[i]), -offset_x*torch.sin(ego_yaw[i])]).to(device)
                elif adjust_plan[scene_idx][agent_id] == 2:
                    # left of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [-offset_y*torch.sin(ego_yaw[i]), offset_y*torch.cos(ego_yaw[i])]).to(device)
                elif adjust_plan[scene_idx][agent_id] == 3:
                    # right of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [offset_y*torch.sin(ego_yaw[i]), -offset_y*torch.cos(ego_yaw[i])]).to(device)
                elif adjust_plan[scene_idx][agent_id] == 4:
                    # two vehicle length ahead of the ego vehicle
                    agent_pos = ego_global[i]+torch.tensor(
                        [2*offset_x*torch.cos(ego_yaw[i]), 2*offset_x*torch.sin(ego_yaw[i])]).to(device)
                agent_pos = GeoUtils.batch_nd_transform_points(
                    agent_pos, obs["agents"]["agent_from_world"][agent_idx])
                agent_pos = agent_pos.tile(T, 1)
                agent_yaw = (ego_yaw[i]-obs["agents"]
                             ["yaw"][agent_idx]).tile(T, 1)
                positions.append(agent_pos)
                yaws.append(agent_yaw)
    agent_obs = dict()
    for k,v in obs["agents"].items():
        agent_obs[k]=v[agent_indices]

    agent_action = OrderedDict(positions=torch.stack(positions,0),yaws=torch.stack(yaws,0))
    agent_obs = TensorUtils.to_numpy(agent_obs)
    agent_action = TensorUtils.to_numpy(agent_action)
    ClosedLoopSimulator.update_agents(
        dataset=env._current_scene_dataset,
        frame_idx=env._frame_index + 1,
        input_dict=agent_obs,
        output_dict=agent_action,
    )


def rollout_episodes(
    env,
    policy,
    num_episodes,
    skip_first_n=1,
    n_step_action=1,
    render=False,
    scene_indices=None,
    start_frame_index_each_episode=None,
    device=None,
    obs_to_torch=True,
    adjust_plan_recipe=None,
    horizon=None,
    seed_each_episode=None,

):
    """
    Rollout an environment for a number of episodes
    Args:
        env (BaseEnv): a base simulation environment (gym-like)
        policy (RolloutWrapper): a policy that controls agents in the environment
        num_episodes (int): number of episodes to rollout for
        skip_first_n (int): number of steps to skip at the begining
        n_step_action (int): number of steps to take between querying models
        render (bool): if True, return a sequence of rendered frames
        scene_indices (tuple, list): (Optional) scenes indices to rollout with
        start_frame_index_each_episode (List): (Optional) which frame to start each simulation episode from,
        device: device to cast observation to
        obs_to_torch: whether to cast observation to torch
        adjust_plan_recipe (dict): (Optional) initialization condition, either a fixed plan or a recipe for random generation
        horizon (int): (Optional) override horizon of the simulation
        seed_each_episode (List): (Optional) a list of seeds, one for each episode

    Returns:
        stats (dict): A dictionary of rollout stats for each episode (metrics, rewards, etc.)
        info (dict): A dictionary of environment info for each episode
        renderings (list): A list of rendered frames in the form of np.ndarray, one for each episode
    """
    stats = {}
    info = {}
    renderings = []
    is_batched_env = isinstance(env, BatchedEnv)
    timers = Timers()
    adjust_plans = list()
    if seed_each_episode is not None:
        assert len(seed_each_episode) == num_episodes
    if start_frame_index_each_episode is not None:
        assert len(start_frame_index_each_episode) == num_episodes
        
    ego_policy = policy.unwrap()["Rollout.ego_policy"]
    trace = list()
    for ei in range(num_episodes):
        if start_frame_index_each_episode is not None:
            start_frame_index = start_frame_index_each_episode[ei]
        else:
            start_frame_index = None
        
        env.reset(scene_indices=scene_indices, start_frame_index=start_frame_index)
        if adjust_plan_recipe is not None:
            if "random_init_plan" in adjust_plan_recipe:
                # recipe provided
                if adjust_plan_recipe["random_init_plan"]:
                    adjust_recipe = adjust_plan_recipe
                    adjust_plan = random_initial_adjust_plan(env,adjust_recipe)
                    
                else:
                    adjust_plan = None
                    adjust_recipe = None
                
            else:
                # explicit plan provided
                adjust_plan = adjust_plan_recipe
        else:
            adjust_plan = None
            adjust_recipe = None
        if adjust_plan is not None:
            env.adjust_scene(adjust_plan)

        if seed_each_episode is not None:
            env.update_random_seed(seed_each_episode[ei])

        done = env.is_done()
        counter = 0
        step_since_last_update = 0
        frames = list()
        while not done:
            if adjust_recipe is not None:
                if step_since_last_update>adjust_recipe["num_frame_per_new_agent"]:
                    for simscene in env._current_scenes:
                        if simscene.scene_ts<simscene.scene.length_timesteps-10:
                            extra_agent = random_placing_neighbors(simscene,1)
                            adjust_plan[simscene.scene.name]["agents"]+=extra_agent
                    env.adjust_scene(adjust_plan)
                    step_since_last_update=0
            timers.tic("step")
            with timers.timed("obs"):
                obs = env.get_observation()
            with timers.timed("to_torch"):
                if obs_to_torch:
                    device = policy.device if device is None else device
                    obs_torch = TensorUtils.to_torch(obs, device=device, ignore_if_unspecified=True)
                else:
                    obs_torch = obs

            with timers.timed("network"):
                action = policy.get_action(obs_torch, step_index=counter)


            if counter < skip_first_n:
                # use GT action for the first N steps to warm up environment state (velocity, etc.)
                gt_action = env.get_gt_action(obs)
                action.ego = gt_action.ego
                action.agents = gt_action.agents
                env.step(action, num_steps_to_take=1, render=False)
                counter += 1
                step_since_last_update+=1
            else:
                with timers.timed("env_step"):
                    ims = env.step(
                        action, num_steps_to_take=n_step_action, render=render
                    )  # List of [num_scene, h, w, 3]
                if render:
                    frames.extend(ims)
                counter += n_step_action
                step_since_last_update += n_step_action
            timers.toc("step")
            # print(timers)

            done = env.is_done()
            
            if horizon is not None and counter >= horizon:
                break
        metrics = env.get_metrics()
        if hasattr(ego_policy,"savetrace") and ego_policy.savetrace:
            trace.append(ego_policy.trace.copy())
            

        for k, v in metrics.items():
            if k not in stats:
                stats[k] = []
            if is_batched_env:  # concatenate by scene
                stats[k] = np.concatenate([stats[k], v], axis=0)
            else:
                stats[k].append(v)

        env_info = env.get_info()
        for k, v in env_info.items():
            if k not in info:
                if isinstance(v,dict):
                    info[k] = dict()
                else:
                    info[k] = list()

            if is_batched_env:
                if isinstance(v,dict):
                    info[k].update(v)
                else:
                    info[k].extend(v)
            else:
                info[k].append(v)
        del env_info
        if hasattr(ego_policy,"reset"):
            ego_policy.reset()
        if render:
            frames = np.stack(frames)
            if is_batched_env:
                # [step, scene] -> [scene, step]
                frames = frames.transpose((1, 0, 2, 3, 4))
            renderings.append(frames)
        if adjust_plan is not None:
            adjust_plans.append(adjust_plan)


    multi_episodes_metrics = env.get_multi_episode_metrics()
    stats.update(multi_episodes_metrics)
    env.reset_multi_episodes_metrics()

    return stats, info, renderings, adjust_plans, trace


class RolloutCallback(pl.Callback):
    """A pytorch-lightning callback function that runs rollouts during training"""
    def __init__(
            self,
            exp_config,
            every_n_steps=100,
            warm_start_n_steps=1,
            verbose=False,
            save_video=False,
            video_dir=None
    ):
        self._every_n_steps = every_n_steps
        self._warm_start_n_steps = warm_start_n_steps
        self._verbose = verbose
        self._exp_cfg = exp_config.clone()
        self._save_video = save_video
        self._video_dir = video_dir
        self.env = None
        self.policy = None
        self._eval_cfg = self._exp_cfg.eval

    def print_if_verbose(self, msg):
        if self._verbose:
            print(msg)

    def _get_env(self, device):
        if self.env is not None:
            return self.env
        if self._eval_cfg.env == "nusc":
            env_builder = EnvNuscBuilder(eval_config=self._eval_cfg, exp_config=self._exp_cfg, device=device)
        elif self._eval_cfg.env == 'l5kit':
            env_builder = EnvL5Builder(eval_config=self._eval_cfg, exp_config=self._exp_cfg, device=device)
        else:
            raise NotImplementedError("{} is not a valid env".format(self._eval_cfg.env))

        env = env_builder.get_env()
        self.env = env
        return self.env

    def _get_policy(self, pl_module: pl.LightningModule):
        if self.policy is not None:
            return self.policy
        policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")

        composer_class = getattr(policy_composers, self._eval_cfg.eval_class)
        composer = composer_class(self._eval_cfg, pl_module.device, ckpt_root_dir=self._eval_cfg.ckpt_root_dir)
        print("Building composer {}".format(self._eval_cfg.eval_class))

        if self._exp_cfg.algo.name == "ma_rasterized":
            policy, _ = composer.get_policy(predictor=pl_module)
        else:
            policy, _ = composer.get_policy(policy=pl_module)

        if self._eval_cfg.policy.pos_to_yaw:
            policy = Pos2YawWrapper(
                policy,
                dt=self._exp_cfg.algo.step_time,
                yaw_correction_speed=self._eval_cfg.policy.yaw_correction_speed
            )

        if self._eval_cfg.env == "nusc":
            rollout_policy = RolloutWrapper(agents_policy=policy)
        elif self._eval_cfg.ego_only:
            rollout_policy = RolloutWrapper(ego_policy=policy)
        else:
            rollout_policy = RolloutWrapper(ego_policy=policy, agents_policy=policy)

        self.policy = rollout_policy
        return self.policy

    def _run_rollout(self, pl_module: pl.LightningModule, global_step: int):
        rollout_policy = self._get_policy(pl_module)
        env = self._get_env(pl_module.device)

        scene_i = 0
        eval_scenes = self._eval_cfg.eval_scenes

        result_stats = None

        while scene_i < len(eval_scenes):
            scene_indices = eval_scenes[scene_i: scene_i + self._eval_cfg.num_scenes_per_batch]
            scene_i += self._eval_cfg.num_scenes_per_batch
            stats, info, renderings,_,_ = rollout_episodes(
                env,
                rollout_policy,
                num_episodes=self._eval_cfg.num_episode_repeats,
                n_step_action=self._eval_cfg.n_step_action,
                render=self._save_video,
                skip_first_n=self._eval_cfg.skip_first_n,
                scene_indices=scene_indices,
                start_frame_index_each_episode=self._eval_cfg.start_frame_index_each_episode,
                seed_each_episode=self._eval_cfg.seed_each_episode,
                horizon=self._eval_cfg.num_simulation_steps
            )

            if result_stats is None:
                result_stats = stats
                result_stats["scene_index"] = np.array(info["scene_index"])
            else:
                for k in stats:
                    result_stats[k] = np.concatenate([result_stats[k], stats[k]], axis=0)
                result_stats["scene_index"] = np.concatenate([result_stats["scene_index"], np.array(info["scene_index"])])

            if self._save_video:
                for ei, episode_rendering in enumerate(renderings):
                    for i, scene_images in enumerate(episode_rendering):
                        video_fn = "{}_{}_{}.mp4".format(global_step, info["scene_index"][i], ei)
                        
                        writer = get_writer(os.path.join(self._video_dir, video_fn), fps=10)
                        print("video to {}".format(os.path.join(self._video_dir, video_fn)))
                        for im in scene_images:
                            writer.append_data(im)
                        writer.close()
        return result_stats

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx, unused=0) -> None:
        should_run = (
            trainer.global_step >= self._warm_start_n_steps
            and trainer.global_step % self._every_n_steps == 0
        )
        if should_run:
            try:
                self.print_if_verbose(
                    "\nStep %i rollout (%i episodes): "
                    % (trainer.global_step, len(self._eval_cfg.eval_scenes))
                )

                stats = self._run_rollout(pl_module, trainer.global_step)
                for k, v in stats.items():
                    if "ttf" in k:  # avoid cluttering the plot
                        continue
                    # Set on_step=True and on_epoch=False to force the logger to log stats at the step
                    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/9772 for explanation
                    pl_module.log(
                        "rollout/" + k, np.mean(v), on_step=True, on_epoch=False
                    )
                    self.print_if_verbose(("rollout/" + k, np.mean(v)))
                self.print_if_verbose("\n")
            except Exception as e:
                print("Rollout failed because:")
                print(e)

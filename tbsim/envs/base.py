import abc


class SimulationException(Exception):
    pass


class BaseEnv(abc.ABC):
    """TODO: Make a Simulator MetaClass"""

    @abc.abstractmethod
    def reset(self, scene_indices=None, start_frame_index=None):
        return

    @abc.abstractmethod
    def reset_multi_episodes_metrics(self):
        pass

    @abc.abstractmethod
    def step(self, action, num_steps_to_take, render):
        return

    @abc.abstractmethod
    def update_random_seed(self, seed):
        return

    @abc.abstractmethod
    def get_metrics(self):
        return

    @abc.abstractmethod
    def get_multi_episode_metrics(self):
        return

    @abc.abstractmethod
    def render(self, actions_to_take):
        return

    @abc.abstractmethod
    def get_info(self):
        return

    @abc.abstractmethod
    def get_observation(self):
        return

    @abc.abstractmethod
    def get_reward(self):
        return

    @abc.abstractmethod
    def is_done(self):
        return

    @abc.abstractmethod
    def get_info(self):
        return


class BatchedEnv(abc.ABC):
    @abc.abstractmethod
    def num_instances(self):
        return

import abc


class Policy(abc.ABC):
    def __init__(self, device, *args, **kwargs):
        self.device = device

    @abc.abstractmethod
    def get_action(self, obs_dict, **kwargs):
        """Predict an action based on the input observation """
        pass

    @abc.abstractmethod
    def eval(self):
        """Set the policy to evaluation mode"""
        pass
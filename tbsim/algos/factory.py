"""Factory methods for creating models"""
from pytorch_lightning import LightningDataModule
from tbsim.configs.base import ExperimentConfig

from tbsim.algos.algos import (
    BehaviorCloning,
    VAETrafficModel,
    DiscreteVAETrafficModel,
    BehaviorCloningGC,
    SpatialPlanner,
    GANTrafficModel,
    BehaviorCloningEC,
    TreeVAETrafficModel,
    SceneTreeTrafficModel,
    ScePTTrafficModel,
    AgentFormerTrafficModel
)

from tbsim.algos.multiagent_algos import (
    MATrafficModel,
)

from tbsim.algos.metric_algos import (
    OccupancyMetric
)



def algo_factory(config: ExperimentConfig, modality_shapes: dict):
    """
    A factory for creating training algos

    Args:
        config (ExperimentConfig): an ExperimentConfig object,
        modality_shapes (dict): a dictionary that maps observation modality names to shapes

    Returns:
        algo: pl.LightningModule
    """
    algo_config = config.algo
    algo_name = algo_config.name

    if algo_name == "bc":
        algo = BehaviorCloning(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "bc_gc":
        algo = BehaviorCloningGC(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "vae":
        algo = VAETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "discrete_vae":
        algo = DiscreteVAETrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "tree":
        algo = SceneTreeTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "bc_ec":
        algo = BehaviorCloningEC(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "spatial_planner":
        algo = SpatialPlanner(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "occupancy":
        algo = OccupancyMetric(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "agent_predictor":
        algo = MATrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "gan":
        algo = GANTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "scept":
        algo = ScePTTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    elif algo_name == "agentformer":
        algo = AgentFormerTrafficModel(algo_config=algo_config, modality_shapes=modality_shapes)
    else:
        raise NotImplementedError("{} is not a valid algorithm" % algo_name)
    return algo

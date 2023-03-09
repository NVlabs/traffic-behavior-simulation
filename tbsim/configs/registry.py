"""A global registry for looking up named experiment configs"""
from tbsim.configs.base import ExperimentConfig

from tbsim.configs.l5kit_config import (
    L5KitTrainConfig,
    L5KitMixedEnvConfig,
    L5KitMixedSemanticMapEnvConfig,
)

from tbsim.configs.nusc_config import (
    NuscTrainConfig,
    NuscEnvConfig
)

from tbsim.configs.algo_config import (
    AgentFormerConfig,
    BehaviorCloningConfig,
    BehaviorCloningECConfig,
    SpatialPlannerConfig,
    BehaviorCloningGCConfig,
    TransformerPredConfig,
    TransformerGANConfig,
    AgentPredictorConfig,
    VAEConfig,
    EBMMetricConfig,
    GANConfig,
    DiscreteVAEConfig,
    TreeAlgoConfig,
    OccupancyMetricConfig,
    UnetConfig,
    ScePTConfig,
    SQPMPCConfig,
)


EXP_CONFIG_REGISTRY = dict()

EXP_CONFIG_REGISTRY["l5_bc"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="l5_bc",
)

EXP_CONFIG_REGISTRY["l5_gan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=GANConfig(),
    registered_name="l5_gan",
)

EXP_CONFIG_REGISTRY["l5_bc_gc"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningGCConfig(),
    registered_name="l5_bc_gc",
)

EXP_CONFIG_REGISTRY["l5_spatial_planner"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="l5_spatial_planner",
)

EXP_CONFIG_REGISTRY["l5_agent_predictor"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="l5_agent_predictor"
)

EXP_CONFIG_REGISTRY["l5_vae"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="l5_vae",
)

EXP_CONFIG_REGISTRY["l5_bc_ec"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=BehaviorCloningECConfig(),
    registered_name="l5_bc_ec",
)

EXP_CONFIG_REGISTRY["l5_discrete_vae"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=DiscreteVAEConfig(),
    registered_name="l5_discrete_vae",
)

EXP_CONFIG_REGISTRY["l5_tree"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=TreeAlgoConfig(),
    registered_name="l5_tree",
)

EXP_CONFIG_REGISTRY["l5_transformer"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=TransformerPredConfig(),
    registered_name="l5_transformer",
)

EXP_CONFIG_REGISTRY["l5_transformer_gan"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedEnvConfig(),
    algo_config=TransformerGANConfig(),
    registered_name="l5_transformer_gan",
)

EXP_CONFIG_REGISTRY["l5_ebm"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=EBMMetricConfig(),
    registered_name="l5_ebm",
)

EXP_CONFIG_REGISTRY["l5_occupancy"] = ExperimentConfig(
    train_config=L5KitTrainConfig(),
    env_config=L5KitMixedSemanticMapEnvConfig(),
    algo_config=OccupancyMetricConfig(),
    registered_name="l5_occupancy"
)

EXP_CONFIG_REGISTRY["nusc_bc"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="nusc_bc"
)

EXP_CONFIG_REGISTRY["nusc_bc_gc"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningGCConfig(),
    registered_name="nusc_bc_gc"
)

EXP_CONFIG_REGISTRY["nusc_spatial_planner"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=SpatialPlannerConfig(),
    registered_name="nusc_spatial_planner"
)

EXP_CONFIG_REGISTRY["nusc_vae"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=VAEConfig(),
    registered_name="nusc_vae"
)

EXP_CONFIG_REGISTRY["nusc_discrete_vae"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=DiscreteVAEConfig(),
    registered_name="nusc_discrete_vae"
)

EXP_CONFIG_REGISTRY["nusc_tree"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=TreeAlgoConfig(),
    registered_name="nusc_tree"
)

EXP_CONFIG_REGISTRY["nusc_diff_stack"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=BehaviorCloningConfig(),
    registered_name="nusc_diff_stack"
)


EXP_CONFIG_REGISTRY["nusc_agent_predictor"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=AgentPredictorConfig(),
    registered_name="nusc_agent_predictor"
)

EXP_CONFIG_REGISTRY["nusc_gan"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=GANConfig(),
    registered_name="nusc_gan"
)

EXP_CONFIG_REGISTRY["nusc_occupancy"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=OccupancyMetricConfig(),
    registered_name="nusc_occupancy"
)
EXP_CONFIG_REGISTRY["nusc_unet"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=UnetConfig(),
    registered_name="nusc_unet"
)

EXP_CONFIG_REGISTRY["nusc_scept"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=ScePTConfig(),
    registered_name="nusc_scept"
)

EXP_CONFIG_REGISTRY["nusc_agentformer"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=AgentFormerConfig(),
    registered_name="nusc_agentformer"
)

EXP_CONFIG_REGISTRY["nusc_MPC"] = ExperimentConfig(
    train_config=NuscTrainConfig(),
    env_config=NuscEnvConfig(),
    algo_config=SQPMPCConfig(),
    registered_name="nusc_MPC"
)

def get_registered_experiment_config(registered_name):
    registered_name = backward_compatible_translate(registered_name)

    if registered_name not in EXP_CONFIG_REGISTRY.keys():
        raise KeyError(
            "'{}' is not a registered experiment config please choose from {}".format(
                registered_name, list(EXP_CONFIG_REGISTRY.keys())
            )
        )
    return EXP_CONFIG_REGISTRY[registered_name].clone()


def backward_compatible_translate(registered_name):
    """Try to translate registered name to maintain backward compatibility."""
    translation = {
        "l5_mixed_plan": "l5_bc",
        "l5_mixed_gc": "l5_bc_gc",
        "l5_ma_rasterized_plan": "l5_agent_predictor",
        "l5_gan_plan": "l5_gan",
        "l5_mixed_ec_plan": "l5_bc_ec",
        "l5_mixed_vae_plan": "l5_vae",
        "l5_mixed_discrete_vae_plan": "l5_discrete_vae",
        "l5_mixed_tree_vae_plan": "l5_tree_vae",
        "nusc_rasterized_plan": "nusc_bc",
        "nusc_mixed_gc": "nusc_bc_gc",
        "nusc_ma_rasterized_plan": "nusc_agent_predictor",
        "nusc_gan_plan": "nusc_gan",
        "nusc_vae_plan": "nusc_vae",
        "nusc_mixed_tree_vae_plan": "nusc_tree",
    }
    if registered_name in translation:
        registered_name = translation[registered_name]
    return registered_name
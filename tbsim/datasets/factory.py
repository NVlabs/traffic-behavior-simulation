"""DataModule / Dataset factory"""
from tbsim.utils.config_utils import translate_l5kit_cfg, translate_trajdata_cfg
from tbsim.datasets.l5kit_datamodules import L5MixedDataModule, L5RasterizedDataModule
from tbsim.datasets.trajdata_datamodules import UnifiedDataModule

def datamodule_factory(cls_name: str, config):
    """
    A factory for creating pl.DataModule.

    Valid module class names: "L5MixedDataModule", "L5RasterizedDataModule"
    Args:
        cls_name (str): name of the datamodule class
        config (Config): an Experiment config object
        **kwargs: any other kwargs needed by the datamodule

    Returns:
        A DataModule
    """
    if cls_name.startswith("L5"):  # TODO: make this less hacky
        l5_config = translate_l5kit_cfg(config)
        datamodule = eval(cls_name)(l5_config=l5_config, train_config=config.train)
    elif cls_name.startswith("Unified"):
        trajdata_config = translate_trajdata_cfg(config)
        datamodule = eval(cls_name)(data_config=trajdata_config, train_config=config.train)
    else:
        raise NotImplementedError("{} is not a supported datamodule type".format(cls_name))
    return datamodule
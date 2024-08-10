from utils import registry


DATASETS = registry.Registry('dataset')

def build_dataset_from_cfg(cfg, default_args = None):
    return DATASETS.build(cfg, default_args = default_args)
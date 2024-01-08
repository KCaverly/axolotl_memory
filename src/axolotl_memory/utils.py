import yaml
from pathlib import Path
from addict import Dict


class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.
    """

    def __missing__(self, key):
        return None

    def __or__(self, other):
        return DictDefault(super().__or__(other))


def load_cfg(config: Path = Path("examples/"), **kwargs):
    # load the config from the yaml file
    with open(config, encoding="utf-8") as file:
        cfg = DictDefault(yaml.safe_load(file))

    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    cfg_keys = cfg.keys()
    for k, _ in kwargs.items():
        # if not strict, allow writing to cfg even if it's not in the yml already
        if k in cfg_keys:
            # handle booleans
            if isinstance(cfg[k], bool):
                cfg[k] = bool(kwargs[k])
            else:
                cfg[k] = kwargs[k]

    return cfg

import argparse
from pathlib import Path

from axolotl_memory.utils import load_cfg
from axolotl_memory.base_model import calculate_memory_base_model


def sizeof_fmt(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate the memory for an axolotl config"
    )
    parser.add_argument(
        "--config", metavar="path", required=True, help="the path to config"
    )

    args = parser.parse_args()
    path = Path(args.config)

    # Load in Config
    cfg = load_cfg(path)

    # Calculate Base Model Size
    base_model = cfg.get("base_model", None)
    if base_model is None:
        raise Exception("'base_model' not available in axolotl config")
    (model_size, largest_layer, empty_model) = calculate_memory_base_model(
        cfg.get("base_model", "unknown")
    )

    print("")
    print(f"Base Model:            {base_model}")
    print(f"Estimated Memory:      {sizeof_fmt(model_size, 'B')}")
    print("")

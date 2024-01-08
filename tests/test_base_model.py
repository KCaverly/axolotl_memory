from pathlib import Path
from axolotl_memory.utils import load_cfg
from axolotl_memory.base_model import calculate_memory_base_model


def test_base_model():
    (model_size, largest_layer, empty_model) = calculate_memory_base_model(
        "bert-base-cased"
    )
    assert model_size > 0
    assert largest_layer[0] < model_size
    assert empty_model is not None


def test_base_model_from_cfg():
    example_path = Path("examples/cerebras/btlm-ft.yml")
    cfg = load_cfg(example_path)

    assert False

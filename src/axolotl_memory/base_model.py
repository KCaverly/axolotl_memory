#
# Alot of the below is borrowed from here:
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py
#
import transformers
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import calculate_maximum_sizes
from huggingface_hub.hf_api import ModelInfo
from transformers import AutoConfig, AutoModel
from typing import Optional, Tuple, List
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


def verify_on_hub(
    repo: str, token: Optional[str] = None
) -> Tuple[str, Optional[ModelInfo]]:
    "Verifies that the model is on the hub and returns the model info."
    try:
        return ("success", model_info(repo, token=token))
    except GatedRepoError:
        return ("gated", None)
    except RepositoryNotFoundError:
        return ("repo", None)


def load_empty_transformers_model(
    model_name: str, model_info, trust_remote_code: bool = True
):
    auto_map = model_info.config.get("auto_map", False)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    with init_empty_weights():
        # remote code could specify a specific `AutoModel` class in the `auto_map`
        constructor = AutoModel
        if isinstance(auto_map, dict):
            value = None
            for key in auto_map.keys():
                if key.startswith("AutoModelFor"):
                    value = key
                    break
            if value is not None:
                constructor = getattr(transformers, value)
        model = constructor.from_config(config, trust_remote_code=trust_remote_code)

        return model


def calculate_memory_base_model(
    model_name: str, token: Optional[str] = None, trust_remote_code: bool = True
) -> Tuple[int, Tuple[int, List[str]], nn.Module]:
    result, model_info = verify_on_hub(model_name, token)

    if result == "gated":
        raise GatedRepoError(
            f"Repo for model `{model_name}` is gated. You must be authenticated to access it. Please run `huggingface-cli login`."
        )
    elif result == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo,"
            " make sure you are authenticated via `huggingface-cli login` and have access."
        )
    else:
        empty_model = load_empty_transformers_model(
            model_name, model_info, trust_remote_code
        )

        total_size, largest_layer = calculate_maximum_sizes(empty_model)

        return total_size, largest_layer, empty_model

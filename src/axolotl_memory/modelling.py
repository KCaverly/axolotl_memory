#
# Alot of the below is borrowed from here:
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py
#
import transformers
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils import calculate_maximum_sizes
from huggingface_hub.hf_api import ModelInfo
from transformers import AutoConfig, AutoModel, LlamaForCausalLM
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
    model_name: str,
    model_info,
    trust_remote_code: bool = True,
):
    auto_map = model_info.config.get("auto_map", False)

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    with init_empty_weights():
        constructor = AutoModel

        if isinstance(auto_map, dict):
            value = None
            print(auto_map)
            for key in auto_map.keys():
                print(key)
                if key.startswith("AutoModelFor"):
                    value = key
                    break
            if value is not None:
                constructor = getattr(transformers, value)
        model = constructor.from_config(
            config,
            trust_remote_code=trust_remote_code,
        )

        return model


def calculate_memory_base_model(
    cfg, trust_remote_code: bool = True, token: Optional[str] = None
) -> Tuple[int, nn.Module]:
    model_name = cfg.get("base_model", None)
    if model_name is None:
        raise Exception("'base_model' is missing from cfg")

    result, model_info = verify_on_hub(model_name, token)

    if cfg.load_in_8bit and cfg.adapter is not None:
        bytes_per_param = 1.0
    elif cfg.load_in_4bit and cfg.adapter is not None:
        bytes_per_param = 0.5
    else:
        bytes_per_param = None

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

        memory = 0
        for param in empty_model.parameters():
            if bytes_per_param == None:
                memory += param.numel() * param.element_size()
            else:
                memory += param.numel() * bytes_per_param

        return memory, empty_model

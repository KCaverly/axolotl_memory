#
# Alot of the below is borrowed from here:
# https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py
#
import torch
import transformers
import torch.nn as nn
import bitsandbytes as bnb
from accelerate import init_empty_weights
from accelerate.utils import calculate_maximum_sizes
from huggingface_hub.hf_api import ModelInfo
from transformers import AutoConfig, AutoModel, LlamaForCausalLM
from typing import Optional, Tuple, List
from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from peft.tuners.lora import QuantLinear
from axolotl_memory.memory import MemoryPrecision, MemoryCategory, MemoryItem


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
            for key in auto_map.keys():
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


def load_base_model(cfg, token: Optional[str] = None):
    model_name = cfg.get("base_model", None)
    if model_name is None:
        raise Exception("'base_model' is missing from cfg")

    result, model_info = verify_on_hub(model_name, token)

    if result == "gated":
        raise GatedRepoError(
            f"Repo for model '{model_name}' is gated. You must be authenticated to access it. Please run `huggingface-cli login`."
        )
    elif result == "repo":
        raise RepositoryNotFoundError(
            f"Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo, make sure you are authenticated via `huggingface-cli login` and have access."
        )
    else:
        return load_empty_transformers_model(
            model_name, model_info, trust_remote_code=True
        )


def calculate_base_model_memory(base_model, cfg):
    if cfg.load_in_8bit and cfg.adapter is not None:
        bytes_per_param = 1.0
        precision = MemoryPrecision.BIT8
    elif cfg.load_in_4bit and cfg.adapter is not None:
        bytes_per_param = 0.5
        precision = MemoryPrecision.BIT4
    else:
        bytes_per_param = None
        precision = None

    base_model_name = cfg.get("base_model")

    memory = 0
    if bytes_per_param is None:
        for param in base_model.parameters():
            memory += param.numel() * param.element_size()

            if param.element_size() == 4:
                if precision is not None and precision != MemoryPrecision.BIT32:
                    precision = MemoryPrecision.MIXED
                else:
                    precision = MemoryPrecision.BIT32

            elif param.element_size() == 2:
                if precision is not None and precision != MemoryPrecision.BIT16:
                    precision = MemoryPrecision.MIXED
                else:
                    precision = MemoryPrecision.BIT16

    else:
        for param in base_model.parameters():
            memory += param.numel() * bytes_per_param

    return MemoryItem(
        MemoryCategory.MODELLING, f"Base Model ({base_model_name})", precision, memory
    )


def get_lora_model(base_model, cfg):
    from peft import LoraConfig, PeftModel, get_peft_model

    lora_target_modules = list(cfg.lora_target_modules or [])

    if cfg.lora_target_linear:
        linear_names = find_all_linear_names(base_model)
        lora_target_modules = list(set(lora_target_modules + linear_names))

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        modules_to_save=cfg.lora_modules_to_save if cfg.lora_modules_to_save else None,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if cfg.lora_model_dir:
        model = PeftModel.from_pretrained(
            base_model, cfg.lora_model_dir, is_trainable=True
        )
    else:
        model = get_peft_model(base_model, lora_config)

    return model


def calculate_lora_adapter_memory(lora_model, cfg):
    if cfg.bf16:
        bytes_per_param = 2.0
        precision = MemoryPrecision.BIT16
    elif cfg.fp16:
        bytes_per_param = 2.0
        precision = MemoryPrecision.BIT16
    elif cfg.fp32:
        bytes_per_param = 4.0
        precision = MemoryPrecision.BIT32
    else:
        bytes_per_param = None

    # Calculate Memory for only Trainable Weights
    memory = 0
    if bytes_per_param is None:
        for param in lora_model.parameters():
            if param.requires_grad:
                memory += param.numel() * param.element_size()

                if param.element_size() == 4:
                    if precision is None:
                        precision = MemoryPrecision.BIT32
                    elif precision != MemoryPrecision.BIT32:
                        precision = MemoryPrecision.MIXED

                elif param.element_size() == 2:
                    if precision is None:
                        precision = MemoryPrecision.BIT16
                    elif precision != MemoryPrecision.BIT16:
                        precision = MemoryPrecision.MIXED

                else:
                    precision = MemoryPrecision.UNKNOWN

    else:
        for param in lora_model.parameters():
            if param.requires_grad:
                memory += param.numel() * bytes_per_param

    return MemoryItem(
        MemoryCategory.MODELLING, f"LORA Adapter (r = {cfg.lora_r})", precision, memory
    )


def find_all_linear_names(model):
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")

    return list(lora_module_names)

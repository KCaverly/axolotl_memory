from collections import defaultdict
from accelerate.utils import (
    calculate_maximum_sizes,
    compute_module_sizes,
    named_module_tensors,
    get_max_layer_size,
)
from axolotl_memory.memory import MemoryItem, MemoryPrecision, MemoryCategory


def calculate_gradient_memory(base_model, cfg) -> MemoryItem:
    # Get Bytes for Trainable Params
    if cfg.bf16:
        trainable_bytes_per_param = 2.0
        precision = MemoryPrecision.BIT16
    elif cfg.fp16:
        trainable_bytes_per_param = 2.0
        precision = MemoryPrecision.BIT16
    elif cfg.fp32:
        trainable_bytes_per_param = 4.0
        precision = MemoryPrecision.BIT32
    else:
        trainable_bytes_per_param = None
        precision = MemoryPrecision.UNKNOWN

    optimizer = cfg.get("optimizer")
    if optimizer in [
        "sgd",
        "adawm_hf",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_torch_xla",
        "adamw_bnb_8bit",
    ]:
        memory = 0
        if trainable_bytes_per_param is None:
            for param in base_model.parameters():
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
            for param in base_model.parameters():
                if param.requires_grad:
                    memory += param.numel() * trainable_bytes_per_param

        return MemoryItem(MemoryCategory.TRAINING, "Gradients", precision, memory)

    else:
        raise NotImplementedError(
            f"{optimizer} not currently implemented for gradient memory calculations"
        )


def calculate_optimizer_state_memory(base_model, cfg) -> MemoryItem:
    optimizer = cfg.get("optimizer")
    if optimizer == "sgd":
        bytes_per_param = 4.0
        state_params = 0
        precision = MemoryPrecision.BIT32
    elif optimizer in [
        "adamw_hf",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_torch_xla",
    ]:
        bytes_per_param = 4.0
        state_params = 2
        precision = MemoryPrecision.BIT32
    elif optimizer in ["adamw_bnb_8bit"]:
        bytes_per_param = 1.0
        state_params = 2
        precision = MemoryPrecision.BIT8
    else:
        raise NotImplementedError(
            f"{optimizer} not currently implemented for optimizer state calculations"
        )

    memory = 0
    for param in base_model.parameters():
        if param.requires_grad:
            memory += param.numel() * state_params * bytes_per_param

    return MemoryItem(
        MemoryCategory.TRAINING, f"Optimizer ({optimizer})", precision, memory
    )


def compute_custom_module_sizes(model, cfg):
    module_sizes = defaultdict(int)

    if cfg.load_in_8bit:
        base_bytes_per_param = 1
    elif cfg.load_in_4bit:
        base_bytes_per_param = 0.5
    else:
        base_bytes_per_param = None

    if cfg.bf16:
        training_bytes_per_param = 2
    elif cfg.fp16:
        training_bytes_per_param = 2
    else:
        training_bytes_per_param = 4

    for name, tensor in named_module_tensors(model, recurse=True):
        if tensor.requires_grad:
            size = tensor.numel() * training_bytes_per_param
        else:
            if base_bytes_per_param is None:
                size = tensor.numel() * tensor.element_size()
            else:
                size = int(tensor.numel() * base_bytes_per_param)

        name_parts = name.split(".")
        for idx in range(len(name_parts) + 1):
            module_sizes[".".join(name_parts[:idx])] += size

    return module_sizes


def calculate_activation_memory(base_model, cfg) -> MemoryItem:
    # The Size of the largest layer in the model * batch_size

    sizes = compute_custom_module_sizes(base_model, cfg)

    no_split_modules = getattr(base_model, "_no_split_modules", None)
    if no_split_modules is None:
        no_split_modules = []

    modules_to_treat = (
        list(base_model.named_parameters(recurse=False))
        + list(base_model.named_children())
        + list(base_model.named_buffers(recurse=False))
    )
    (largest_layer_size, _) = get_max_layer_size(
        modules_to_treat, sizes, no_split_modules
    )

    if (cfg.fp16 or cfg.bf16 or cfg.fp32) and (cfg.load_in_8bit or cfg.load_in_4bit):
        precision = MemoryPrecision.MIXED
    elif cfg.fp16 or cfg.bf16:
        precision = MemoryPrecision.BIT16
    elif cfg.fp32:
        precision = MemoryPrecision.BIT32
    else:
        precision = MemoryPrecision.UNKNOWN

    return MemoryItem(
        MemoryCategory.TRAINING,
        f"Activations",
        precision,
        largest_layer_size,
    )

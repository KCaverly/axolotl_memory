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

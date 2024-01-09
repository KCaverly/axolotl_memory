def calculate_memory_for_optimizer(base_model, cfg):
    """Return the memory requirements for the optimizer in bytes"""

    optimizer = cfg.get("optimizer")
    if optimizer == "sgd":
        if cfg.load_in_4bit and cfg.adapter is not None:
            bytes_per_param = 0.5
        elif cfg.load_in_8bit and cfg.adapter is not None:
            bytes_per_param = 1.0
        else:
            bytes_per_param = None

        memory = 0
        if bytes_per_param is None:
            for param in base_model.parameters():
                if param.requires_grad:
                    memory += param.numel() * param.element_size()
        else:
            for param in base_model.parameters():
                if param.requires_grad:
                    memory += param.numel() * bytes_per_param

        return memory

    elif optimizer in [
        "adamw_hf",
        "adamw_torch",
        "adamw_torch_fused",
        "adamw_torch_xla",
    ]:
        # M_model * 3
        # Gradients = M_model
        # First and Second Moment = M_model * 2

        if cfg.load_in_4bit and cfg.adapter is not None:
            bytes_per_param = 0.5
        elif cfg.load_in_8bit and cfg.adapter is not None:
            bytes_per_param = 1.0
        else:
            bytes_per_param = None

        memory = 0
        if bytes_per_param is None:
            for param in base_model.parameters():
                if param.requires_grad:
                    memory += param.numel() * param.element_size()
        else:
            for param in base_model.parameters():
                if param.requires_grad:
                    memory += param.numel() * bytes_per_param

        return memory * 3

    elif optimizer in ["adamw_bnb_8bit"]:
        memory = 0
        for param in base_model.parameters():
            if param.requires_grad:
                memory += param.numel() * 1

        return memory * 3

    else:
        raise NotImplementedError(f"'{optimizer}' memory estimate not implemented")

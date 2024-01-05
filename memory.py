import yaml
from pathlib import Path
from typing import Dict

from transformers import AutoConfig, AutoModel
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
    #
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


def get_bit(cfg: Dict) -> int:
    if cfg.get("fp16", False):
        return 4
    elif cfg.get("bf16", False):
        return 4
    elif cfg.get("float16", False):
        return 4
    else:
        return 8


def calculate_embed_memory(vocab_size, hidden_size, bit):
    return vocab_size * hidden_size * bit


def calculate_block_memory(num_layers, hidden_size):
    return (bit * num_layers * hidden_size) * (13 + (12 * hidden_size))


def calculate_attention_memory(
    num_layers, batch_size, sequence_len, hidden_size, num_attention_heads
):
    return (num_layers * batch_size * sequence_len * hidden_size) * (
        67 + ((9 * num_attention_heads * sequence_len) / hidden_size)
    )


if __name__ == "__main__":
    configs = [
        Path("examples/openllama-3b/config.yml"),
        # Path("examples/cerebras/btlm-ft.yml"), # Estimate off by 14%
        # Path("examples/code-llama/7b/lora.yml"),
        # Path("examples/code-llama/13b/lora.yml"),
    ]
    for cfg_path in configs:
        cfg = load_cfg(cfg_path)
        model_config = AutoConfig.from_pretrained(
            cfg.get("base_model"), trust_remote_code=True
        )

        bit = get_bit(cfg)
        vocab_size = model_config.vocab_size
        hidden_size = model_config.hidden_size
        num_layers = model_config.num_hidden_layers
        batch_size = 1
        sequence_len = cfg.sequence_len
        num_attention_heads = model_config.num_attention_heads

        print("\n")
        print(f"bit:            {bit}")
        print(f"vocab_size:     {vocab_size}")
        print(f"hidden_size:    {hidden_size}")
        print(f"num layers:     {num_layers}")
        print(f"batch size:     {batch_size}")

        embed_memory = calculate_embed_memory(vocab_size, hidden_size, bit)
        block_memory = calculate_block_memory(num_layers, hidden_size)
        # attention_memory = calculate_attention_memory(
        #     1, batch_size, sequence_len, hidden_size, num_attention_heads
        # )

        print("\n")
        print(f"embed memory:         {embed_memory}")
        print(f"block memory:         {block_memory}")
        # print(f"attention memory:     {attention_memory}")

        estimate_memory = embed_memory + block_memory  # + attention_memory

        # Estimate from Model Itself
        model = AutoModel.from_pretrained(cfg.get("base_model"), trust_remote_code=True)
        model_memory = 0
        for param in model.parameters():
            step_memory = param.numel() * param.element_size()
            model_memory += step_memory

        for layer in model.children():
            print(layer)
            layer_memory = 0
            for param in layer.parameters():
                step_memory = param.numel() * param.element_size()
                layer_memory += step_memory

            print(f"layer memory: {layer_memory}")

        print("\n")
        print(f"estimate memory:      {estimate_memory}")
        print(f"model memory:         {model_memory}")
        print(
            f"difference:           {(estimate_memory - model_memory) / model_memory}"
        )

        # print(cfg)
        # print(model_config)
        # print(model)

        del model

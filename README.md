# axolotl_memory
Small experiments on estimating transformer memory from axolotl config files.

This is primarily a learning project, in which I am looking to predict the memory consumption of a LLM from an [axolotl]() config file alone.
The hope is that if we can get a reasonable solution for this, we can move it over to the `axolotl` project directly. [Issue Here](https://github.com/OpenAccess-AI-Collective/axolotl/issues/848)
## How to Use

Simply pass the axolotl config file to the main script as a '--config' file path to estimate the size in memory.

```
python main.py --config examples/code-llama/34b/lora.yml
```

Would return:

```
┌───────────────────────────────────────────────────────────────┐
│                        Estimate Memory                        │
├───────────────────────────────────────────────────┬───────────┤
│ Modelling                                         │           │
├───────────────────────────────────────────────────┼───────────┤
│  Base Model (codellama/CodeLlama-34b-hf - 8bit)   │  31.2GiB  │
│  LORA Adapter                                     │  207.8MiB │
├───────────────────────────────────────────────────┬───────────┤
│ Training                                          │           │
├───────────────────────────────────────────────────┼───────────┤
│  Gradients                                        │  207.8MiB │
│  Optimizer: adamw_bnb_8bit                        │  415.5MiB │
└───────────────────────────────────────────────────┴───────────┘
```

### Functionality

Borrowing from [here](https://tinkerd.net/blog/machine-learning/distributed-training/#measuring-the-four-sources-of-memory-consumption), we group memory requirements into three broad buckets:

#### 1. Model Memory

The memory required to load the model into memory. Includes base model, quantized or unquantized, and peft adapters.

| Model Base | Base Model | 4bit | 8bit | LORA | QLORA | GPTQ | GPTQ w/Flash Attn | flash attn | xformers attn |
| ---------- | ---------- | --- | --- | -------- | - | --- | --- | --- | --- |
| Llama      | ✔️          | ✔️  | ✔️  | ✔️       | :x: | :x: | :x: | :x: | :x: |

#### 2. Gradient & Optimizer Memory

The memory required for a single backward pass of the model.

| Optimizer | Basic |
| --- | --- |
| sgd | ✔️ |
| adamw_hf| ✔️ |
| adamw_torch | ✔️ |
| adamw_torch_fused | ✔️ |
| adamw_apex_fused | :x: |
| adamw_anyprecision | :x: |
| adafactor | :x: |
| adamw_bnb_8bit | ✔️ |

#### 3. Activation Memory

The required memory to do a forward pass of the model.

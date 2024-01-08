# axolotl_memory
Small experiments on estimating transformer memory from axolotl config files.

This is primarily a learning project, in which I am looking to predict the memory consumption of a LLM from an [axolotl]() config file alone.
The hope is that if we can get a reasonable solution for this, we can move it over to the `axolotl` project directly. [Issue Here](https://github.com/OpenAccess-AI-Collective/axolotl/issues/848)
## How to Use

Simply pass the axolotl config file to the main script as a '--config' file path to estimate the size in memory.

`python main.py --config examples/code-llama/7b/lora.yml`

Would return:

```
Base Model:            codellama/CodeLlama-7b-hf
Estimated Memory:      25.1GiB
```

### Definitions

Borrowing from [here](https://tinkerd.net/blog/machine-learning/distributed-training/#measuring-the-four-sources-of-memory-consumption), we group memory requirements into three broad buckets:

1. Model Memory

The memory required in bytes for storing the model on its own.

2. Gradient & Optimizer Memory

This is the required memory, to calculate the necessary gradients and update the model weights.

3. Activation Memory

This is the required memory, to calculate a forward pass of the model.

### Roadmap

1. Estimate model memory from Transformers base model.
- Leverage [accelerate's estimate-memory](https://github.com/huggingface/accelerate/blob/main/src/accelerate/commands/estimate.py) command to grab an empty version of the Transformers model, without the weights loaded into memory, and calculate the size of the base model.

2. Calculate basic memory requirements for standard optimizers.
- SGD
- AdamW
- AdamW - 8bit
- ...

### How can we test this?

I think a reasonable goal is to estimate memory without 10% to start, and move forward from there.
To start, we should be able to profile memory on cpu for loading the base model into memory.

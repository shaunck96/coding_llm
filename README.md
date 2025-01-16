# Titan Memory-Integrated Llama-2 for Code Generation

This document outlines how to train a **Llama-2-based** Large Language Model (LLM) with a **custom “Titan” memory module**, which persistently maintains and updates a learned representation of prior context. The overall goal is to build a **coding assistant** that can generate code while leveraging an external memory to adapt over time. 

**Note**: This project is a proof-of-concept to demonstrate how one might integrate persistent neural memory into a causal language model. Achieving **state-of-the-art code generation** (comparable to tools like Copilot, Code Llama, StarCoder, etc.) requires extensive data, compute resources, and additional fine-tuning strategies. Nonetheless, this repository shows a framework to get started.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Key Features](#key-features)  
3. [Usage & Setup](#usage--setup)  
   1. [Environment Setup](#environment-setup)  
   2. [Running the Notebook or Script](#running-the-notebook-or-script)  
4. [Components Explanation](#components-explanation)  
   1. [NeuralMemory](#1-neuralmemory)  
   2. [LlamaTitanModel](#2-llamatitanmodel)  
   3. [Training Pipeline](#3-training-pipeline)  
5. [Customization & Extensions](#customization--extensions)  
6. [Performance & Scaling](#performance--scaling)  
7. [Limitations](#limitations)  
8. [License & Acknowledgments](#license--acknowledgments)

---

## Project Structure

```
.
├── train.py    # Example script/notebook showcasing the full training flow
├── README.md                     # This README
└── requirements.txt              # (Optional) If you choose to define your environment dependencies
```

- **TitanMemoryLLM_Training.py** (or a notebook version) contains the end-to-end training pipeline.  
- **README.md** (this file) explains the conceptual flow and usage.  
- **requirements.txt** is optional; you might define the environment dependencies there if you want to replicate the environment easily.

---

## Key Features

1. **Llama-2 Causal Language Modeling**  
   - Uses `LlamaForCausalLM` from Hugging Face Transformers to provide next-token prediction for code or text.

2. **Titan Memory Module**  
   - A custom `NeuralMemory` class keeps a persistent buffer of size `MEMORY_SIZE`.  
   - Each forward pass, it computes a “surprise” measure based on hidden-state differences and updates this buffer.  
   - Integrates memory-augmented representations back into the LLM’s hidden states.

3. **Hugging Face Accelerate**  
   - Handles distributed training, mixed precision, or multi-GPU setups with minimal changes to the code.  
   - Simplifies device handling and gradient synchronization.

4. **MLflow Integration**  
   - Tracks hyperparameters, training loss, and other metrics.  
   - Saves experiment logs so you can compare across runs.

5. **Built for Code Datasets**  
   - Integrates with code-related datasets like CodeSearchNet, The Stack, or Apps from Hugging Face.  
   - Demonstrates typical steps for text preprocessing and labeled data (if relevant).

---

## Usage & Setup

### Environment Setup

1. **Python Environment**  
   - Python 3.8+ is recommended (though 3.9+ also works).  
   - You need a GPU with substantial VRAM for training a 7B+ model (e.g., NVIDIA A100 with 40GB, or multiple GPUs using Accelerate).

2. **Install Dependencies**  
   - If running in a Databricks notebook, you can prepend each cell with `%pip install ...` or run directly:
     ```bash
     pip install torch torchvision transformers datasets tokenizers accelerate deepspeed mlflow
     ```
   - Alternatively, define them in `requirements.txt` and run:
     ```bash
     pip install -r requirements.txt
     ```

3. **Model Checkpoint**  
   - You must have access to `meta-llama/Llama-2-7b-hf` (or a similar Llama-2 checkpoint) on Hugging Face. This typically requires acceptance of the license terms.

### Running the Notebook or Script

1. **Databricks Notebook**  
   - Upload the `.py` or `.ipynb` file to your Databricks workspace.  
   - Attach to a GPU cluster (with at least one A100, ideally).  
   - Update any configuration paths (e.g., `CHECKPOINT_DIR`, `LOGGING_DIR`) to point to your preferred storage.  
   - Run all cells in order.

2. **Local / Server Training**  
   - Make sure you have GPUs accessible locally or on a remote server with CUDA.  
   - Clone or copy this repository.  
   - Modify the paths (e.g., `CHECKPOINT_DIR`) in the config class.  
   - Run:
     ```bash
     python TitanMemoryLLM_Training.py
     ```

> **Tip**: If you are memory-constrained, reduce `BATCH_SIZE` or `MAX_LENGTH` to avoid out-of-memory (OOM) errors.

---

## Components Explanation

### 1. `NeuralMemory`

- **Location**: Defined inside the script/notebook under `class NeuralMemory(nn.Module)`.  
- **Function**:  
  - Maintains a persistent buffer (`self.memory`) with shape `(MEMORY_SIZE, HIDDEN_DIM)`.  
  - Projects the model’s hidden states into a memory space, computes a “surprise” measure by comparing them to the persistent buffer, and updates the buffer.  
  - Returns a memory-augmented hidden state that is later combined with the model’s own representation.

- **Key Methods**:  
  - `forward(hidden_states)`:  
    1. Projects `hidden_states` into memory space.  
    2. Computes difference (`L2 norm`) with the persistent memory.  
    3. Calculates “surprise” as average difference across memory slots.  
    4. Updates memory outside the gradient flow using a weighted average of past memory and the new states.  
    5. Returns the final memory-augmented representation.

- **Design Note**: This approach is simplistic. For large-scale usage, you may need more advanced gating or a recurrent-like memory for stable gradient-based learning.

### 2. `LlamaTitanModel`

- **Location**: `class LlamaTitanModel(nn.Module)`.  
- **Function**:  
  - Wraps `LlamaForCausalLM` (the standard Llama-2 architecture for next-token prediction).  
  - Integrates the `NeuralMemory` after obtaining the base hidden states from Llama.  
  - Applies an optional `MultiheadAttention` to blend the original hidden states with the memory outputs.  
  - Finally, passes the resulting representation to the Llama language modeling head to get output logits.

- **Key Steps**:  
  1. Get hidden states from `self.llama_model.model(...)`.  
  2. Pass them to `self.memory_module(...)`.  
  3. Combine via `self.memory_attention(...)`.  
  4. Generate logits via `self.llama_model.lm_head(...)`.  
  5. Compute a causal language modeling loss if `labels` are provided.

### 3. Training Pipeline

1. **Data Loading**:  
   - Uses `load_dataset` from Hugging Face to fetch code-related datasets (`code_search_net`, `bigcode/the-stack`, `codeparrot/apps`).  
   - Concatenates them.  
   - Tokenizes up to `MAX_LENGTH` tokens, sets `labels = input_ids` for causal LM.

2. **Accelerate Setup**:  
   - Creates an `Accelerator` instance.  
   - Prepares model, optimizer, and dataloader for distributed or mixed-precision training.

3. **Forward & Backward Pass**:  
   - Each iteration, calls `model(input_ids, attention_mask, labels)`.  
   - Loss is backpropagated, and steps are accumulated until `GRADIENT_ACCUMULATION_STEPS` is reached.  
   - Optimizer and scheduler are stepped, and gradients are zeroed out.

4. **MLflow Logging**:  
   - Logs hyperparameters at the start of the run.  
   - Logs training loss at each epoch step.  
   - (Optional) Add custom logs or validation metrics as needed.

5. **Checkpointing**:  
   - After all epochs, calls `accelerator.unwrap_model(model)` to get the base model.  
   - Saves it to `CHECKPOINT_DIR` using `save_pretrained`.

---

## Customization & Extensions

1. **Partial Freezing**:  
   - Uncomment lines in `LlamaTitanModel.__init__` to freeze early layers if you only want to train the memory and final layers.

2. **Different Datasets**:  
   - Change `config.DATASET_NAMES` to your target code or text datasets.  
   - Ensure the `tokenize_function` reflects the correct column names (e.g., `'code'`, `'text'`, etc.).

3. **Evaluation & Validation**:  
   - Create a separate validation set or reserve part of your dataset.  
   - Periodically evaluate perplexity or generation quality on the validation set.

4. **Memory Mechanism**:  
   - Replace the simplistic memory update logic with a more advanced approach (e.g., gating, trainable memory parameters, or recurrent-like memory).

5. **Model Size**:  
   - Switch to “meta-llama/Llama-2-13b-hf” or “meta-llama/Llama-2-70b-hf” if you have the necessary hardware resources and you want a bigger foundation.  
   - For smaller experiments or prototyping, consider “meta-llama/Llama-2-7b-hf” or even a smaller code-specific base.

---

## Performance & Scaling

- **GPU Requirements**:  
  - A single 7B model with `MAX_LENGTH=1024` can use 20–30 GB of VRAM or more, depending on batch size.  
  - To handle multi-GPU setups, ensure you use `Accelerate` in distributed mode, or run on a cluster (e.g., Databricks with multiple A100s).

- **Gradient Checkpointing**:  
  - If you encounter out-of-memory errors, enable `model.llama_model.gradient_checkpointing_enable()` or reduce `MAX_LENGTH`/`BATCH_SIZE`.

- **Throughput**:  
  - Training code LLMs can be slow. Consider further optimizations or distributed training strategies.

---

## Limitations

1. **Memory Implementation**:  
   - The `NeuralMemory` uses a simple in-place update that is mostly outside the standard gradient flow. This might limit the effectiveness of the memory.  
   - A more robust approach could use a differentiable memory with trainable parameters and gating mechanisms.

2. **Data Volume**:  
   - Public code datasets can be extremely large (e.g., The Stack). For production usage, you may want to stream the data or use dataset subsets.

3. **Evaluation & RLHF**:  
   - This script does not implement a reward model or RLHF, which are often crucial for refining code generation quality and instruction-following behavior.

4. **Licensing & Access**:  
   - Access to Llama-2 checkpoints requires agreement to Meta’s license. Ensure you have permissions.

---

## License & Acknowledgments

- **Llama-2 Model**: Provided by Meta under its [Llama-2 license terms](https://ai.meta.com/resources/meta-llama-license/).  
- **Transformers & Datasets**: By [Hugging Face](https://huggingface.co/), under Apache 2.0.  
- **MLflow**: An open-source platform for managing ML lifecycles.  
- **Accelerate**: An open-source library by Hugging Face for easy distributed and mixed-precision training.

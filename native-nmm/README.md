# Native-Multimodal Model in JAX

This repository implements a native-multimodal model using Python and JAX (with Flax/Linen). It provides a complete pipeline from tokenizer training and data preparation to pretraining, supervised fine-tuning (SFT), and inference.

## Features

- **Store LLaVA data**: Download and store LLaVA image-description data locally.
- **Tokenizer Training**: Train a custom tokenizer on Fineweb data.
- **Pretraining**: Native multimodal pretraining on mixed datasets (Fineweb + LLaVA).
- **Supervised Fine-Tuning (SFT)**: Instruction tuning for chat capabilities.
- **Inference**: Support for both text-only causal inference and multimodal chat.
- **Server**: A simple chat server implementation.

## Project Structure

- `src/`: Core modules (models, data processing, utils).
- `scripts/`: Executable scripts for training and inference.
- `configs/`: Model configurations.

## Installation

This project uses `uv` for dependency management.

```bash
uv venv
uv pip install .
```

## Usage

### 1. Train Tokenizer
Train a tokenizer using the Fineweb dataset.
```bash
uv run scripts/train_tokenizer.py
```

### 2. Prepare Data
Download and store LLaVA image-description data locally to accelerate training.
```bash
uv run scripts/save_llava_mid_local.py
```

### 3. Pretraining
Perform pretraining using the mixed dataset. Model architecture is defined in `configs/model.yaml`.

```bash
uv run scripts/train_native_pretrain.py
```

> **Note**: Hyperparameters are currently hardcoded in the script. Modify `scripts/train_native_pretrain.py` if needed.

**(Optional) Verify Pretraining:**
Run text-only causal inference to check the pretrained model.
```bash
uv run scripts/inference_text_causal.py
```

### 4. Supervised Fine-Tuning (SFT)
Fine-tune the pretrained model for chat capabilities.
```bash
uv run scripts/train_sft_chat.py
```

### 5. Inference & Demo
Run multimodal chat inference with the fine-tuned model.
```bash
uv run scripts/inference_mm_chat.py
```

**Launch Chat Server:**
Start a server backend for the chat interface.
```bash
uv run scripts/server_chat.py
```

## References
![https://arxiv.org/abs/2509.23661](LLaVA-OneVision-1.5)
![https://arxiv.org/abs/2504.07951](Scaling Laws for Native Multimodal Models)
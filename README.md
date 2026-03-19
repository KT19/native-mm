# 🌌 Native Multimodal Model (NMM)

A complete, end-to-end implementation of a **Native Multimodal Model** built with **JAX/Flax**, featuring a high-performance training pipeline and a modern React-based chat interface.

---

## Overview

This project provides a full-stack solution for training and interacting with multimodal large language models. Unlike traditional modular approaches, this repository focuses on **native** multimodal integration, where images and text are processed within a unified architecture.

### Key Features

- **JAX-Powered Backend**: Extreme performance and scalability using JAX and Flax/Linen.
- **Full Pipeline**: Includes scripts for tokenizer training, data preparation (LLaVA), pretraining, and Supervised Fine-Tuning (SFT).
- **Native Multimodal**: Unified processing of text and image data.
- **Modern Chat UI**: A sleek, responsive frontend built with React, Vite, and TailwindCSS for real-time interaction.

---

## Project Structure

The repository is organized into two main components:

| Component                        | Description                                                                      |
| :------------------------------- | :------------------------------------------------------------------------------- |
| [**`native-nmm/`**](./native-mm) | The core machine learning codebase (JAX models, training scripts, data loaders). |
| [**`chat-ui/`**](./chat-ui)      | The React-based frontend application for interacting with the model.             |

---

## Getting Started

### 1. Backend Setup (`native-mm/`)

```bash
cd native-mm
uv venv
uv pip install .
```

Refer to the [**`native-mm` documentation**](./native-mm/README.md) for detailed steps on:

- Training your own tokenizer.
- Preparing the LLaVA dataset.
- Running Pretraining & SFT.
- Starting the Chat Server.

### 2. Frontend Setup (`chat-ui/`)

```bash
cd chat-ui
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`. Make sure the backend server (from `native-mm`) is running to enable chat functionality.

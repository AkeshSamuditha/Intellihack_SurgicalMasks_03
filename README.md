# Intellihack_SurgicalMasks_03
This repo is created as part of the intellihack first round Task 04

## Project Overview

This repository contains the code and models developed for the first round of the Intellihack competition, specifically for Task 04 which focuses on surgical mask detection and analysis.

## Models

The project utilizes several key models:

1. **BAAI/bge-large-en-v1.5** - This embedding model was selected for its excellent performance in text representation and similarity search. It offers multiple dimension options (768, 512, 256, 128, 64) through its matryoshka architecture, providing flexibility in balancing performance and computational efficiency.

2. **Qwen/Qwen2.5-3B-Instruct** - We chose this instruction-tuned model for its strong performance on domain-specific tasks. The 3B parameter size offers a good balance between accuracy and resource requirements, making it suitable for both training and inference.

## Data Processing

The repository includes scripts for:
- Loading and preprocessing datasets (train_dataset.json and val_dataset.json)
- Creating corpus and query datasets for embedding evaluation
- Training and fine-tuning procedures for both embedding and language models

## Implementation Details

- **Embedding Fine-tuning** (`embedding_ft.py`): Implements embedding model training and evaluation using SentenceTransformer with a focus on information retrieval tasks
- **Custom Training** (`custom_train.py`): Provides a customized training pipeline for the Qwen model using Unsloth for optimization

## Environment Setup

The project uses a Python virtual environment with all dependencies listed in the requirements. The `.gitignore` is configured to exclude environment files, caches, and other non-essential files.

## Inference

For inference, please refer to the included notebook which provides a step-by-step guide to:
1. Load the fine-tuned models
2. Process input data
3. Generate embeddings for query and context matching
4. Run inference with the Qwen model on matched context-query pairs
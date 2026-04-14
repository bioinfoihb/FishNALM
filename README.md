# FishNALM

**FishNALM: A Fish-Specific Foundation DNA Language Model for Fish Genomes**

> Placeholder Hugging Face links are intentionally left in this README. Replace them after you create your repositories.

## Overview

FishNALM is a fish-specific foundation DNA language model framework for fish genomes. It is designed to learn biologically relevant sequence representations from large-scale fish genomic sequences and to support downstream genomic prediction tasks such as promoter prediction, transcription factor binding-site prediction, histone-mark prediction, splice-site prediction, and variant-effect analysis.

In our study, the FishNALM family includes four pretrained models built from two data scales and two model scales:

- **FishNALM-8** and **FishNALM-8L**: pretrained on 8 core cyprinid fish genomes
- **FishNALM-20** and **FishNALM-20L**: pretrained on 20 diverse fish genomes

FishNALM uses a **BERT-style encoder-only Transformer** with **masked language modeling (MLM)** as the pretraining objective. The sequence tokenizer adopts a **6-mer + BPE** hybrid strategy. In the current public codebase, we focus on the **BERT-based pipeline** used in our project.

## Highlights

- Fish-specific DNA foundation language models
- BERT-style encoder architecture with MLM pretraining
- 6-mer + BPE hybrid tokenization
- Support for **pretraining**, **fine-tuning**, and **inference**
- Designed for fish genomic prediction tasks and downstream biological analysis

## Model summary

| Model | Pretraining data | Approx. parameters | Architecture |
| --- | --- | ---: | --- |
| FishNALM-8 | 8 fish genomes | ~90M | BERT-style encoder |
| FishNALM-8L | 8 fish genomes | ~310M | BERT-style encoder |
| FishNALM-20 | 20 fish genomes | ~90M | BERT-style encoder |
| FishNALM-20L | 20 fish genomes | ~310M | BERT-style encoder |

Common settings used in FishNALM:

- Tokenization: **6-mer + BPE**
- Vocabulary size: **4863**
- Maximum input length: **512 tokens**
- Objective: **masked language modeling (MLM)**

## Repository structure

```text
FishNALM/
├── model_pretrain.py
├── model_finetune.py
├── model_inference.py
├── split_dataset.py
├── pdllib/
├── examples_pretrain/
│   ├── test_sequences.txt
│   └── test_pretrain.sh
├── examples_finetune/
│   ├── pro_tata_train.csv
│   ├── pro_tata_val.csv
│   ├── pro_tata_test.csv
│   └── test_finetune.sh
├── examples_inference/
│   ├── test_inference.fa
│   └── test_inference.sh
└── requirements.txt
```

## Environment

We recommend using **conda** to create the environment.

### 1. Create environment

```bash
conda create -n fishnalm python=3.11
conda activate fishnalm
```

### 2. Install PyTorch

For GPU training or inference, install the PyTorch version that matches your CUDA environment.

Example for CUDA 12.1:

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cu121
```

If you only want CPU inference:

```bash
pip install 'torch<2.4' --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install other dependencies

```bash
git clone --recursive https://github.com/YOUR_GITHUB_USERNAME/FishNALM.git
cd FishNALM
pip install -r requirements.txt
```

Optional: if you want to enable FlashAttention in supported environments, you can additionally install:

```bash
pip install flash-attn
```

## Pretrained models

Replace the links below with your final Hugging Face repositories after release.

| Model | Link |
| --- | --- |
| FishNALM-8 | `https://huggingface.co/YOUR_HF_USERNAME/FishNALM-8` |
| FishNALM-8L | `https://huggingface.co/YOUR_HF_USERNAME/FishNALM-8L` |
| FishNALM-20 | `https://huggingface.co/YOUR_HF_USERNAME/FishNALM-20` |
| FishNALM-20L | `https://huggingface.co/YOUR_HF_USERNAME/FishNALM-20L` |

If `git-lfs` is installed, the models can be downloaded with:

```bash
git lfs install
git clone https://huggingface.co/YOUR_HF_USERNAME/FishNALM-8
```

If downloading from Hugging Face is slow, you can optionally use a mirror:

```bash
export HF_ENDPOINT="https://hf-mirror.com"
huggingface-cli download YOUR_HF_USERNAME/FishNALM-8
```

## Pretraining

FishNALM supports MLM pretraining with `model_pretrain.py`.

### Input format

Prepare a plain text file for pretraining, with **one DNA sequence per line**.

Example:

```text
ACGTACGTACGT...
TTGCAAGCTTAA...
```

A minimal example script is available in `examples_pretrain/test_pretrain.sh`.

### Example command

```bash
torchrun --nproc_per_node=4 model_pretrain.py \
  --model_name_or_path /path/to/initial_model_or_checkpoint \
  --train_data examples_pretrain/test_sequences.txt \
  --is_mlm True \
  --from_scratch True \
  --bf16 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --logging_steps 100 \
  --save_steps 1000 \
  --output_dir pretrain_output \
  --model_max_length 512 \
  --max_steps 200
```

### Main arguments

- `--model_name_or_path`: initial model path or checkpoint path
- `--train_data`: training sequences file
- `--is_mlm`: whether to use masked language modeling
- `--from_scratch`: whether to train from scratch
- `--per_device_train_batch_size`: batch size per device
- `--gradient_accumulation_steps`: gradient accumulation steps
- `--learning_rate`: learning rate
- `--warmup_ratio`: warmup ratio
- `--bf16`: enable bf16 mixed precision
- `--logging_steps`: interval for logging
- `--save_steps`: interval for saving checkpoints
- `--output_dir`: output directory
- `--model_max_length`: maximum input length in tokens
- `--max_steps`: maximum training steps

## Fine-tuning

Fine-tuning is performed with `model_finetune.py`.

### Input format

Fine-tuning data should be provided as a CSV file with at least the following columns:

```csv
sequence,label
ACGT...,1
TTGC...,0
```

A minimal example is available in `examples_finetune/`.

### Example command

```bash
torchrun --nproc_per_node=4 model_finetune.py \
  --model_name_or_path /path/to/pretrained_checkpoint \
  --tokenizer_path /path/to/pretrained_checkpoint \
  --train_data examples_finetune/pro_tata_train.csv \
  --eval_data examples_finetune/pro_tata_val.csv \
  --test_data examples_finetune/pro_tata_test.csv \
  --train_task classification \
  --labels "Not;Yes" \
  --output_dir finetune_output \
  --run_name promoter_tata_test \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.05 \
  --model_max_length 512 \
  --bf16 \
  --load_best_model_at_end \
  --metric_for_best_model matthews_correlation
```

### Main arguments

- `--model_name_or_path`: pretrained model or checkpoint path
- `--tokenizer_path`: tokenizer path
- `--train_data`: training dataset
- `--eval_data`: validation dataset
- `--test_data`: test dataset
- `--train_task`: task type, such as classification
- `--labels`: class names separated by `;`
- `--output_dir`: output directory
- `--run_name`: run name
- `--per_device_train_batch_size`: training batch size per device
- `--per_device_eval_batch_size`: evaluation batch size per device
- `--learning_rate`: learning rate
- `--num_train_epochs`: number of epochs
- `--warmup_ratio`: warmup ratio
- `--model_max_length`: maximum input length in tokens
- `--metric_for_best_model`: model selection metric

## Inference

Inference is performed with `model_inference.py` using a fine-tuned model.

### Input format

You can provide:

- a single sequence from the command line
- a FASTA/plain text file with one sequence per line
- a CSV/TSV file containing a `sequence` column

### Example command

```bash
python model_inference.py \
  -m /path/to/finetuned_model \
  -f examples_inference/test_inference.fa \
  -o predict_results.txt
```

### Common arguments

- `-m`: fine-tuned model path
- `-s`: input DNA sequence
- `-f`: input file containing sequences
- `-o`: output file for predictions

## Example downstream tasks

Based on our current project, FishNALM can be adapted to tasks such as:

- promoter prediction
- transcription factor binding-site prediction
- histone-mark prediction
- splice-site prediction
- zero-shot variant effect analysis

## Paper

- *FishNALM: A Fish-Specific Foundation DNA Language Model for Fish Genomes*

## License

This project is distributed under the terms described in the `LICENSE` file.

## Contact

**Xiao-Qin Xia**  
Institute of Hydrobiology, Chinese Academy of Sciences  
Email: xqxia@ihb.ac.cn  
Email: bioinfoihb@ihb.ac.cn

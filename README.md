# 🌐 LoRA-MT: Multilingual Translation via LoRA Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Model: Hy-MT2](https://img.shields.io/badge/base_model-Hy--MT2--7B-green.svg)](https://huggingface.co/tencent/Hy-MT2-7B)

English | [简体中文](README_CN.md)

LoRA + DoRA fine-tuning on Tencent's Hy-MT2-7B, covering **150+ language directions across 5 domains** with only a **110MB** adapter.

## ✨ Features

- 📦 **Lightweight**: LoRA adapter only 110MB (base model downloaded separately)
- 🌍 **Multilingual**: 150+ language directions, including 138 low-resource languages
- 🗣️ **Dialect Support**: Cantonese → Mandarin (placeholder protection for character preservation)
- ⚖️ **Professional**: Legal, business, medical, and diplomatic text precision
- 🔁 **Word Order**: SVO/SOV/VSO/framing structures → natural target language
- ⚡ **Plug & Play**: One-command inference script included

## 📊 Results

| Metric             | Base (Hy-MT2-7B) | Fine-Tuned       |
| ------------------ |:----------------:|:----------------:|
| Eval Loss          | —                | **1.026**        |
| Cantonese→Mandarin | Average          | ✅ Accurate       |
| Swahili→English    | Average          | ✅ Understandable |
| Legal en→zh        | Average          | ✅ Precise        |

> Detailed comparison: `docs/验证结果_基座.txt` and `docs/验证结果_微调.txt`

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/Adobiz/mt-lora.git
cd mt-lora
pip install -r requirements.txt
```

### Download Base Model

```bash
huggingface-cli download tencent/Hy-MT2-7B --local-dir ./Hy-MT2-7B
```

### Inference

```bash
# Single translation
python submission/inference.py --text "Hello world" --lang en-zh

# Interactive mode
python submission/inference.py --interactive

# Batch translation
python submission/inference.py --input test.txt --lang en-zh
```

## 🔧 Training

### Data Preparation

```bash
# Collect corpora from OPUS/UN/smol_dataset, format as alpaca JSON
python scripts/preprocess_cantonese.py        # Cantonese protection
python scripts/convert_to_llamafactory.py     # Format conversion
```

### Launch Training

```bash
# Single GPU
python scripts/train.py \
    --model_path ./Hy-MT2-7B \
    --data_dir ./data \
    --output_dir ./output_7b \
    --batch_size 8 --lora_rank 32
```

### Evaluation

```bash
# 20 test cases × 5 challenges
python scripts/validate.py --model ./Hy-MT2-7B --adapter ./output/final_lora

# BLEU scoring
python scripts/eval_metrics.py --model ./Hy-MT2-7B --adapter ./output/final_lora
```

## 📁 Project Structure

```
mt-lora/
├── README.md                          # English
├── README_CN.md                       # 简体中文
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── train.py                       # Training script
│   ├── validate.py                    # Validation (20 cases)
│   ├── eval_metrics.py                # BLEU evaluation
│   ├── preprocess_cantonese.py        # Cantonese protection
│   └── convert_to_llamafactory.py     # Alpaca conversion
├── submission/
│   ├── inference.py                   # Inference script
│   └── README.md
├── docs/
│   ├── 验证结果_基座.txt
│   ├── 验证结果_微调.txt
│   └── 微调技术要点总结.md
└── data/                              # Training data (prepare separately)
```

## 🧠 Technical Highlights

| #   | Technique                | Description                                        |
|:---:| ------------------------ | -------------------------------------------------- |
| 1   | **LoRA + DoRA**          | rank=32, 0.37% trainable params                    |
| 2   | **Mixed Training**       | 5 datasets injected in one pass                    |
| 3   | **Cantonese Protection** | 20 dialect characters preserved during conversion  |
| 4   | **Dynamic Prompts**      | Auto-generated instructions for 138 language pairs |
| 5   | **Framework-Free**       | Pure transformers+PEFT, zero extra deps            |

> See `docs/微调技术要点总结.md` for details.

## 📚 Data Sources

| Source                  | Sentence Pairs | License |
| ----------------------- | --------------:| ------- |
| UN Parallel Corpus v1.0 | 62.56M         | UN      |
| WMT19 News Translation  | 4M             | Public  |
| OPUS-100 (Helsinki-NLP) | 1.01M          | CC      |
| news_commentary v16     | 0.4M           | WMT     |
| smol_dataset            | 0.75M          | Open    |
| cantonese_zh            | 24K            | Public  |

## 🖥️ Hardware

| Scenario   | GPU                 | VRAM  |
| ---------- | ------------------- |:-----:|
| Inference  | Any CUDA GPU        | 15 GB |
| Train 7B   | 4090D / 4090 / 3090 | 24 GB |
| Train 1.8B | 4060+               | 8 GB  |

## 📝 Citation

Base model: [Tencent Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B)

```bibtex
@software{mt-lora,
  title = {LoRA-MT: Multilingual Translation via LoRA Fine-Tuning},
  year = {2026},
  url = {https://github.com/Adobiz/LoRA-MT}
}
```

## 📄 License

MIT — see [LICENSE](LICENSE)

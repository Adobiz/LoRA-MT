# 🌐 LoRA-MT: 多语种翻译模型 LoRA 微调

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Model: Hy-MT2](https://img.shields.io/badge/base_model-Hy--MT2--7B-green.svg)](https://huggingface.co/tencent/Hy-MT2-7B)

基于腾讯 Hy-MT2-7B 的 LoRA + DoRA 微调，覆盖 **150+ 语言方向、5 大领域**，adapter 仅 **110MB**。

## ✨ 特性

- 📦 **轻量化**：LoRA 微调，adapter 仅 110MB（基座模型可独立下载）
- 🌍 **多语种**：150+ 语言方向，含 138 种低资源语言
- 🗣️ **方言支持**：粤语→普通话（占位符保护法防止特征字丢失）
- ⚖️ **专业文本**：法律、商务、医学、外交领域精准翻译
- 🔁 **多语序**：SVO/SOV/VSO/框形结构 → 自然中文
- ⚡ **即开即用**：提供推理脚本，一行命令运行

## 📊 效果对比

| 指标         | 基座 (Hy-MT2-7B) | 微调后       |
| ---------- |:--------------:|:---------:|
| BLEU (Avg) | 71.01          | **72.78** |
| 粤语→普通话     | 一般             | ✅ 准确      |
| 斯瓦希里语→英文   | 一般             | ✅ 可理解     |
| 法律文本 en→zh | 一般             | ✅ 术语精准    |

> 详细对比见 `docs/验证结果_基座.txt` 和 `docs/验证结果_微调.txt`

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/Adobiz/lora-mt.git
cd lora-mt
pip install -r requirements.txt
```

### 下载基座模型

```bash
huggingface-cli download tencent/Hy-MT2-7B --local-dir ./Hy-MT2-7B
```

### 推理

```bash
# 单条翻译
python submission/inference.py --text "Hello world" --lang en-zh

# 交互模式
python submission/inference.py --interactive

# 批量翻译(lang en-zh|英译中)
python submission/inference.py --input test.txt --lang en-zh
```

## 🔧 训练

### 准备数据

```bash
# 从 OPUS/UN/smol_dataset 等来源收集语料
# 按 alpaca 格式组织为 JSON 文件
# 运行预处理（粤语保护等）
python scripts/preprocess_cantonese.py
python scripts/convert_to_llamafactory.py
```

### 启动训练

```bash
# 单 GPU
python scripts/train.py \
    --model_path ./Hy-MT2-7B \
    --data_dir ./data \
    --output_dir ./output_7b \
    --batch_size 8 --lora_rank 32

# 多 GPU
torchrun --nproc_per_node=4 scripts/train.py ...
```

### 验证

```bash
# 20 条测试用例 × 5 大挑战
python scripts/validate.py --model ./Hy-MT2-7B --adapter ./output/final_lora

# BLEU 评估
python scripts/eval_metrics.py --model ./Hy-MT2-7B --adapter ./output/final_lora
```

## 📁 项目结构

```
lora-mt/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── train.py                     # 训练脚本
│   ├── validate.py                  # 20 用例验证
│   ├── eval_metrics.py              # BLEU 评估
│   ├── preprocess_cantonese.py      # 粤语占位符保护
│   └── convert_to_llamafactory.py   # alpaca 格式转换
├── submission/                      # 提交交付物
│   ├── inference.py                 # 推理脚本
│   ├── README.md                    # 使用说明
│   └── final_lora/                  # LoRA 权重 (需下载)
├── docs/
│   ├── 验证结果_基座.txt
│   ├── 验证结果_微调.txt
│   └── 微调技术要点总结.md
└── data/                            # 训练数据
```

## 🧠 技术要点

| #   | 技术              | 说明                         |
|:---:| --------------- | -------------------------- |
| 1   | **LoRA + DoRA** | rank=32, 仅训 0.37% 参数       |
| 2   | **混合训练**        | 5 数据集一次性注入，防灾难性遗忘          |
| 3   | **粤语占位符保护**     | 20 个特征字在繁转简时被保护            |
| 4   | **动态指令**        | 138 种语言方向自动生成正确 prompt     |
| 5   | **去框架化**        | 直用 transformers+PEFT，零额外依赖 |

> 详见 `docs/微调技术要点总结.md`

## 📚 训练数据来源

| 来源                      | 句对数    | 许可    |
| ----------------------- | ------:| ----- |
| UN Parallel Corpus v1.0 | 6256 万 | UN 许可 |
| WMT19 News Translation  | 400 万  | 公开    |
| OPUS-100 (Helsinki-NLP) | 101 万  | CC    |
| news_commentary v16     | 40 万   | WMT   |
| smol_dataset            | 75 万   | 开源    |
| cantonese_zh            | 2.4 万  | 公开    |

## 🖥️ 硬件需求

| 场景      | GPU                 | 显存    |
| ------- | ------------------- |:-----:|
| 推理      | 任意 CUDA GPU         | 15 GB |
| 训练 7B   | 4090D / 4090 / 3090 | 24 GB |
| 训练 1.8B | 4060+               | 8 GB  |

## 📝 引用

基座模型：[Tencent Hy-MT2](https://huggingface.co/tencent/Hy-MT2-7B)

```bibtex
@software{lora-mt,
  title = {LoRA-MT: Multilingual Translation via LoRA Fine-Tuning},
  year = {2026},
  url = {https://github.com/Adobiz/lora-mt}
}
```

## 📄 许可

MIT License — 见 [LICENSE](LICENSE)

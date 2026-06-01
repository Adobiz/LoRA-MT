# 多语种翻译微调模型 — 使用说明

## 环境

```bash
pip install torch transformers peft
```

## 下载基座模型

```bash
pip install -U huggingface_hub
huggingface-cli download tencent/Hy-MT2-7B --local-dir ./Hy-MT2-7B
```

## 推理

```bash
# 单条
python inference.py --text "Hello world" --lang en-zh

# 交互
python inference.py --interactive

# 批量
python inference.py --input test.txt --lang en-zh
```

## 支持的语言方向

| 参数 | 方向 |
|------|------|
| en-zh | 英文→中文 |
| zh-en | 中文→英文 |
| yue-zh | 粤语→普通话 |
| fr-zh | 法文→中文 |
| de-zh | 德文→中文 |
| es-zh | 西班牙文→中文 |
| ja-zh | 日文→中文 |
| ru-zh | 俄文→中文 |
| ar-zh | 阿拉伯文→中文 |

## 文件

- `final_lora/` — LoRA adapter 权重（110MB）
- `inference.py` — 推理脚本

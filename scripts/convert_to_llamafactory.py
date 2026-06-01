#!/usr/bin/env python3
"""
数据格式转换 v2：平行句对 → LLaMA-Factory alpaca 格式
修复：支持一对多领域分配 + 全局采样上限
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# ========== 配置 ==========
CORPUS_DIR = Path("D:/huggingface_cache/AI_Tuning/原始语料")
LLAMA_DATA = Path("D:/huggingface_cache/AI_Tuning/LLaMA-Factory/data")
DATASET_INFO = LLAMA_DATA / "dataset_info.json"

random.seed(42)

# 语言名称
LANG_NAMES = {
    "zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文",
    "fr": "法文", "de": "德文", "es": "西班牙文", "ru": "俄文",
    "ar": "阿拉伯文", "pt": "葡萄牙文", "th": "泰文", "vi": "越南文",
    "id": "印尼文", "ms": "马来文", "hi": "印地文",
    "it": "意大利文", "nl": "荷兰文", "cs": "捷克文",
    "yue": "粤语", "nan": "闽南语", "hak": "客家话",
    "bo": "藏语", "ug": "维吾尔文",
}

# 全局每领域上限
DOMAIN_CAP = {
    "专业翻译/法律医疗":   200000,
    "商务翻译/商务报告":   150000,
    "通用翻译/新闻类":     200000,
    "通用翻译/日常对话类": 150000,
    "口语翻译/方言口语":   100000,
}

# 采样计划：(文件名关键词, 领域, 每个文件采样数, instruction提示词)
SAMPLING_PLAN = [
    # === 专业法律 ===
    ("un_corpus_en-zh", "专业翻译/法律医疗", 50000,
     "将以下联合国法律文件中英文文本翻译成中文，保持法律术语的准确性和正式语体"),
    ("un_corpus_es-zh", "专业翻译/法律医疗", 30000,
     "将以下联合国法律文件中西班牙文文本翻译成中文，保持法律术语的准确性和正式语体"),
    ("un_corpus_fr-zh", "专业翻译/法律医疗", 30000,
     "将以下联合国法律文件中法文文本翻译成中文，保持法律术语的准确性和正式语体"),
    ("un_corpus_ru-zh", "专业翻译/法律医疗", 30000,
     "将以下联合国法律文件中俄文文本翻译成中文，保持法律术语的准确性和正式语体"),

    # === 商务报告（同一份UN数据，不同领域） ===
    ("un_corpus_en-zh", "商务翻译/商务报告", 40000,
     "将以下商务报告中英文文本翻译成中文，保持专业术语和商务语体"),
    ("un_corpus_es-zh", "商务翻译/商务报告", 20000,
     "将以下商务报告中西班牙文文本翻译成中文，保持专业术语和商务语体"),
    ("un_corpus_fr-zh", "商务翻译/商务报告", 20000,
     "将以下商务报告中法文文本翻译成中文，保持专业术语和商务语体"),
    ("un_corpus_ru-zh", "商务翻译/商务报告", 20000,
     "将以下商务报告中俄文文本翻译成中文，保持专业术语和商务语体"),

    # === 新闻 ===
    ("news_commentary_en-zh", "通用翻译/新闻类", 50000,
     "将以下新闻文本从英文翻译成中文，保持新闻语体的简洁准确"),
    ("news_commentary_de-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从德文翻译成中文"),
    ("news_commentary_es-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从西班牙文翻译成中文"),
    ("news_commentary_fr-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从法文翻译成中文"),
    ("news_commentary_ru-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从俄文翻译成中文"),
    ("news_commentary_ar-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从阿拉伯文翻译成中文"),
    ("news_commentary_it-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从意大利文翻译成中文"),
    ("news_commentary_ja-zh", "通用翻译/新闻类", 570,
     "将以下新闻文本从日文翻译成中文"),
    ("news_commentary_pt-zh", "通用翻译/新闻类", 10000,
     "将以下新闻文本从葡萄牙文翻译成中文"),
    ("wmt19_zh-en", "通用翻译/新闻类", 30000,
     "将以下新闻文本从中文翻译成英文"),

    # === 日常对话 ===
    ("opus100_en-zh", "通用翻译/日常对话类", 80000,
     "将以下日常文本从英文翻译成中文，使用自然流畅的表达"),
    ("opus100_de-zh", "通用翻译/日常对话类", 2000,
     "将以下文本从德文翻译成中文"),
    ("opus100_fr-zh", "通用翻译/日常对话类", 2000,
     "将以下文本从法文翻译成中文"),
    ("opus100_ar-zh", "通用翻译/日常对话类", 2000,
     "将以下文本从阿拉伯文翻译成中文"),
    ("opus100_ru-zh", "通用翻译/日常对话类", 2000,
     "将以下文本从俄文翻译成中文"),
    ("opus100_nl-zh", "通用翻译/日常对话类", 2000,
     "将以下文本从荷兰文翻译成中文"),

    # === 方言口语 ===
    ("cantonese", "口语翻译/方言口语", 24000,
     "将以下粤语口语表达转换为标准中文普通话"),

    # === 低资源语言 → 方言口语（动态指令：根据实际语言方向生成） ===
    ("smol_", "口语翻译/方言口语", 500, ""),
]


# ========== 工具函数 ==========

def load_pairs(meta_path):
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    base = str(meta_path).replace(".meta.json", "")
    src_file = Path(base + "." + meta["src_lang"])
    tgt_file = Path(base + "." + meta["tgt_lang"])
    if not src_file.exists() or not tgt_file.exists():
        return [], meta
    with open(src_file, encoding="utf-8") as fs, open(tgt_file, encoding="utf-8") as ft:
        pairs = [(s.strip(), t.strip()) for s, t in zip(fs, ft) if s.strip() and t.strip()]
    return pairs, meta


def build_instruction(src_lang, tgt_lang, prompt):
    src_name = LANG_NAMES.get(src_lang, src_lang)
    tgt_name = LANG_NAMES.get(tgt_lang, tgt_lang)
    if prompt:
        return f"{prompt}："
    return f"将以下{src_name}文本翻译成{tgt_name}："


# ========== 主流程 ==========

def main():
    all_metas = list(CORPUS_DIR.glob("**/*.meta.json"))
    print(f"发现 {len(all_metas)} 个语料文件")

    # 收集所有条目（暂不写文件，先放内存去重）
    domain_buckets = defaultdict(list)  # domain -> [(instruction, input, output)]

    for mp in all_metas:
        pairs, meta = load_pairs(mp)
        if not pairs:
            continue

        # 找出所有匹配的采样计划（一个文件可能匹配多个）
        matches = []
        for keyword, domain, target_n, prompt in SAMPLING_PLAN:
            if keyword in mp.name:
                matches.append((domain, target_n, prompt))

        if not matches:
            continue  # 跳过不匹配的文件

        src_l, tgt_l = meta["src_lang"], meta["tgt_lang"]

        for domain, target_n, prompt in matches:
            # 采样
            if len(pairs) > target_n:
                sampled = random.sample(pairs, target_n)
            else:
                sampled = pairs

            instruction = build_instruction(src_l, tgt_l, prompt)

            for src_text, tgt_text in sampled:
                domain_buckets[domain].append((instruction, src_text, tgt_text))

    # 应用全局上限 + 去重
    print("\n采样结果（上限裁剪后）：")
    total = 0
    final_buckets = {}

    for domain, items in sorted(domain_buckets.items()):
        cap = DOMAIN_CAP.get(domain, 50000)
        if len(items) > cap:
            items = random.sample(items, cap)

        # 按 output 去重
        seen = set()
        unique = []
        for inst, inp, out in items:
            key = out[:100]  # 用 output 前100字符做去重key
            if key not in seen:
                seen.add(key)
                unique.append((inst, inp, out))

        random.shuffle(unique)
        final_buckets[domain] = unique
        print(f"  {domain}: {len(unique):,} 条 (上限 {cap:,})")
        total += len(unique)

    print(f"\n总计: {total:,} 条")

    # 按领域分别保存 JSON
    new_datasets = {}

    for domain, items in sorted(final_buckets.items()):
        safe_name = domain.replace("/", "_").replace("翻译_", "")
        json_path = LLAMA_DATA / f"translation_{safe_name}.json"

        output = [{"instruction": i, "input": inp, "output": out} for i, inp, out in items]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        mb = json_path.stat().st_size / 1024 / 1024
        print(f"  保存: {json_path.name} ({len(output):,} 条, {mb:.1f} MB)")

        new_datasets[f"translation_{safe_name}"] = {
            "file_name": f"translation_{safe_name}.json"
        }

    # 更新 dataset_info.json
    print(f"\n更新 dataset_info.json ...")
    with open(DATASET_INFO, "r", encoding="utf-8") as f:
        info = json.load(f)
    # 移除旧条目
    info = {k: v for k, v in info.items() if not k.startswith("translation_")}
    info.update(new_datasets)
    with open(DATASET_INFO, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"已注册 {len(new_datasets)} 个数据集:")
    for name in sorted(new_datasets.keys()):
        print(f"  ✓ {name}")

    print(f"\n{'='*60}")
    print("LLaMA-Factory 训练配置建议：")
    print(f"{'='*60}")
    print(f"  基座模型:    tencent/Hy-MT2-7B (或 1.8B 快速验证)")
    print(f"  微调算法:    LoRA")
    print(f"  LoRA rank:   32")
    print(f"  LoRA alpha:  64")
    print(f"  训练轮数:    2-3 epochs")
    print(f"  学习率:      2e-4")
    print(f"  批大小:      8 (根据显存调整)")
    print(f"  截断长度:    256")
    print(f"  总样本量:    {total:,}")
    print(f"\nLLaMA-Factory WebUI 中选择数据集:")
    for name in sorted(new_datasets.keys()):
        print(f"    ✓ {name}")


if __name__ == "__main__":
    main()

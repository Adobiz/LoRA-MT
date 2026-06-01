"""
翻译质量评估：BLEU + COMET
用法:
  python eval_metrics.py --model <模型路径> --adapter <adapter路径> --device cuda
"""
import re
import torch
import json
import argparse
import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============ 测试数据 ============
# (方向, 原文, 参考答案)
TEST_DATA = [
    # en→zh
    ("en→zh", "The weather is beautiful today.", "今天天气很好。"),
    ("en→zh", "The committee approved the proposal unanimously after a brief discussion.",
     "委员会经过简短讨论后一致通过了该提案。"),
    ("en→zh", "The parties shall resolve all disputes through arbitration.",
     "双方应通过仲裁解决所有争议。"),
    ("en→zh", "The company reported a 15% increase in quarterly revenue.",
     "公司报告季度收入增长了15%。"),
    # zh→en
    ("zh→en", "今天天气很好，我打算去公园散步。",
     "The weather is beautiful today and I plan to go for a walk in the park."),
    ("zh→en", "人工智能正在改变我们的生活方式。",
     "Artificial intelligence is changing the way we live."),
    # fr→zh
    ("fr→zh", "La liberté consiste à pouvoir faire tout ce qui ne nuit pas à autrui.",
     "自由在于能够做不损害他人的一切事情。"),
    # de→zh
    ("de→zh", "Die Welt ist alles, was der Fall ist.",
     "世界是一切发生的事情。"),
    # es→zh
    ("es→zh", "El cambio climático es uno de los mayores desafíos de nuestro tiempo.",
     "气候变化是我们这个时代最大的挑战之一。"),
    # ja→zh
    ("ja→zh", "私は昨日図書館で借りた本を友達にあげました。",
     "我把我昨天在图书馆借的书给了朋友。"),
    # yue→zh
    ("yue→zh", "你哋聽日會唔會去睇戲啊？",
     "你们明天会不会去看电影啊？"),
    ("yue→zh", "佢喺嗰度等咗好耐都冇等到。",
     "他在那里等了好久都没等到。"),
]

PROMPTS = {
    "en→zh": "将以下英文文本翻译成中文：\n{text}\n\n翻译结果：",
    "zh→en": "Translate the following Chinese text into English:\n{text}\n\nTranslation:",
    "yue→zh": "将以下粤语口语表达转换为标准中文普通话：\n{text}\n\n普通话：",
    "fr→zh": "将以下法文文本翻译成中文：\n{text}\n\n翻译结果：",
    "de→zh": "将以下德文文本翻译成中文：\n{text}\n\n翻译结果：",
    "es→zh": "将以下西班牙文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ja→zh": "将以下日文文本翻译成中文：\n{text}\n\n翻译结果：",
}


def load_model(model_path, adapter_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, device_map=device if device == "cuda" else "cpu",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def translate_batch(model, tokenizer, items, device):
    """批量翻译"""
    results = []
    for lang, src, ref in items:
        template = PROMPTS.get(lang, "将以下文本翻译成中文：\n{text}\n\n翻译结果：")
        prompt = template.format(text=src)
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen_ids = out[0][len(inputs["input_ids"][0]):]
        hyp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        # 截断：只取第一个完整句子（遇到句号、感叹号、问号处停止）
        match = re.match(r'(.*?[。.！!？?])', hyp)
        if match:
            hyp = match.group(1)
        results.append((lang, src, hyp, ref))
    return results


def compute_bleu(hypotheses, references):
    """计算 sacreBLEU"""
    try:
        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        score = bleu.corpus_score(hypotheses, [references])
        return score.score
    except ImportError:
        print("  ⚠ sacrebleu 未安装，pip install sacrebleu")
        return None


def compute_comet(data, device, batch_size=8):
    """计算 COMET (使用 wmt22-comet-da 模型)"""
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)
        comet_model = comet_model.to(device)

        comet_data = []
        for lang, src, hyp, ref in data:
            comet_data.append({
                "src": src,
                "mt": hyp,
                "ref": ref,
            })

        seg_scores, sys_score = comet_model.predict(comet_data, batch_size=batch_size)
        return sys_score, seg_scores
    except ImportError:
        print("  ⚠ unbabel-comet 未安装，pip install unbabel-comet")
        return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="基座模型路径")
    parser.add_argument("--adapter", type=str, default="", help="LoRA adapter 路径")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./评估结果.txt")
    args = parser.parse_args()

    print("=" * 60)
    print("翻译质量评估: BLEU + COMET")
    print("=" * 60)

    # 1. 加载模型
    print("\n[1/4] 加载模型...")
    t0 = time.time()
    model, tokenizer = load_model(args.model, args.adapter, args.device)
    print(f"  耗时: {time.time()-t0:.0f}s")

    # 2. 翻译
    print(f"\n[2/4] 翻译 {len(TEST_DATA)} 条...")
    results = translate_batch(model, tokenizer, TEST_DATA, args.device)

    # 打印逐条
    for lang, src, hyp, ref in results:
        print(f"  [{lang}] {src[:50]}...")
        print(f"    预测: {hyp[:80]}")
        print(f"    参考: {ref[:80]}")

    # 3. BLEU
    print(f"\n[3/4] 计算 BLEU...")
    # 按语言分开算
    lang_groups = {}
    for lang, src, hyp, ref in results:
        if lang not in lang_groups:
            lang_groups[lang] = {"hyps": [], "refs": []}
        lang_groups[lang]["hyps"].append(hyp)
        lang_groups[lang]["refs"].append(ref)

    bleu_scores = {}
    all_hyps = []
    all_refs = []
    for lang, g in lang_groups.items():
        score = compute_bleu(g["hyps"], g["refs"])
        if score is not None:
            bleu_scores[lang] = score
            print(f"  {lang}: BLEU={score:.2f}")
        all_hyps.extend(g["hyps"])
        all_refs.extend(g["refs"])

    overall_bleu = compute_bleu(all_hyps, all_refs)
    if overall_bleu is not None:
        print(f"  综合 BLEU: {overall_bleu:.2f}")

    # 4. COMET
    print(f"\n[4/4] 计算 COMET...")
    sys_score, seg_scores = compute_comet(results, args.device)
    if sys_score is not None:
        print(f"  系统级 COMET: {sys_score:.4f}")

    # 保存
    lines = []
    lines.append("=" * 60)
    lines.append("翻译质量评估结果")
    lines.append("=" * 60)
    lines.append("")
    lines.append("--- BLEU ---")
    for lang, score in bleu_scores.items():
        lines.append(f"  {lang}: {score:.2f}")
    if overall_bleu is not None:
        lines.append(f"  综合: {overall_bleu:.2f}")
    lines.append("")
    lines.append(f"--- COMET (wmt22-comet-da) ---")
    if sys_score is not None:
        lines.append(f"  系统级: {sys_score:.4f}")
    lines.append("")
    lines.append("--- 逐条详情 ---")
    for i, (lang, src, hyp, ref) in enumerate(results):
        lines.append(f"[{i+1}] {lang}")
        lines.append(f"  原文: {src}")
        lines.append(f"  预测: {hyp}")
        lines.append(f"  参考: {ref}")
        if seg_scores is not None and i < len(seg_scores):
            lines.append(f"  COMET: {seg_scores[i]:.4f}")
        lines.append("")

    report = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n✅ 报告: {args.output}")


if __name__ == "__main__":
    main()

"""
翻译模型全面验证脚本
支持基座模型和微调后模型对比

用法:
  # 测试微调后模型
  python validate.py --adapter ./output/final_lora --device cuda

  # 测试基座模型
  python validate.py --device cuda

  # 输出到指定文件
  python validate.py --adapter ./output/final_lora --output result.txt
"""
import torch
import time
import json
import os
import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============ 测试用例 ============
# (挑战类别, 标签, 语言方向, 原文, 期望翻译)
TEST_CASES = [
    # ── 一、低资源语言（xx → en 方向：模型只需解码英文）──
    ("低资源语言", "斯瓦希里语(sw→en)", "sw→en",
     "Karibu katika kijiji chetu. Tunafurahi kukuona hapa.",
     "Welcome to our village. We are happy to see you here."),
    ("低资源语言", "阿非利卡语(af→en)", "af→en",
     "Die kinders speel buite in die tuin met hul hond.",
     "The children are playing outside in the garden with their dog."),
    ("低资源语言", "塔加洛语(tl→en)", "tl→en",
     "Magandang umaga sa inyong lahat. Kumusta kayo ngayong araw?",
     "Good morning to all of you. How are you today?"),
    ("低资源语言", "毛利语(mi→en)", "mi→en",
     "Kei te haere au ki te toa ki te hoko kai.",
     "I am going to the store to buy food."),

    # ── 二、方言口语 ──
    ("方言口语", "粤语→普通话(1)", "yue→zh",
     "你哋聽日會唔會去睇戲啊？",
     "你们明天会不会去看电影啊？"),
    ("方言口语", "粤语→普通话(2)", "yue→zh",
     "我哋而家去食飯，你嚟唔嚟啊？",
     "我们现在去吃饭，你来不来啊？"),
    ("方言口语", "粤语→普通话(3)", "yue→zh",
     "佢喺嗰度等咗好耐都冇等到，真係好嬲。",
     "他在那里等了好久都没等到，真的很生气。"),
    ("方言口语", "粤语→普通话(4)", "yue→zh",
     "你食咗飯未呀？我啱啱先收工。",
     "你吃饭了没呀？我刚刚才下班。"),

    # ── 三、专业文本 ──
    ("专业文本", "法律(en→zh)", "en→zh",
     "The parties hereto shall resolve any dispute arising from or in connection "
     "with this Agreement through friendly consultation. Should consultation fail, "
     "either party may submit the dispute to arbitration in accordance with the "
     "rules of the International Chamber of Commerce.",
     "双方应通过友好协商解决因本协议引起或与本协议有关的任何争议。若协商不成，"
     "任何一方均可根据国际商会规则将争议提交仲裁。"),
    ("专业文本", "外交(fr→zh)", "fr→zh",
     "Les États membres réaffirment leur attachement aux principes de la "
     "Charte des Nations Unies et au respect des droits de l'homme.",
     "各成员国重申其对《联合国宪章》原则的坚持以及对尊重人权的承诺。"),
    ("专业文本", "商务(en→zh)", "en→zh",
     "The company's EBITDA margin improved by 320 basis points year-over-year, "
     "driven by operational efficiency gains and a favorable product mix shift "
     "toward higher-margin segments.",
     "公司EBITDA利润率同比提升320个基点，得益于运营效率提升及产品结构向高利润板块的有利转变。"),
    ("专业文本", "医学(de→zh)", "de→zh",
     "Die klinische Studie zeigte eine statistisch signifikante Reduktion "
     "der Mortalitätsrate bei Patienten mit chronischer Herzinsuffizienz.",
     "临床试验显示慢性心力衰竭患者的死亡率出现了统计学显著的下降。"),

    # ── 四、语序差异 ──
    ("语序差异", "英语SVO(en→zh)", "en→zh",
     "The committee approved the proposal unanimously after a brief discussion.",
     "委员会经过简短讨论后一致通过了该提案。"),
    ("语序差异", "德语框形结构(de→zh)", "de→zh",
     "Der Professor hat das von seinen Studenten vorgeschlagene Experiment "
     "nach langen Überlegungen schließlich doch durchgeführt.",
     "经过长时间考虑，教授最终还是进行了学生们提出的实验。"),
    ("语序差异", "阿拉伯语VSO(ar→zh)", "ar→zh",
     "قرأ الطالب الكتاب في المكتبة أمس.",
     "学生昨天在图书馆读了书。"),
    ("语序差异", "俄语(ru→zh)", "ru→zh",
     "Искусственный интеллект стремительно меняет подходы к обработке "
     "естественного языка и машинному переводу.",
     "人工智能正在迅速改变自然语言处理和机器翻译的方法。"),

    # ── 五、通用场景 ──
    ("通用场景", "日常对话(en→zh)", "en→zh",
     "A: How was your weekend? B: Not bad, I just stayed home and binge-watched "
     "that new series everyone's talking about.",
     "周末过得不错，待在家刷了那部人人都在讨论的新剧。"),
    ("通用场景", "新闻财经(en→zh)", "en→zh",
     "The European Central Bank decided to keep interest rates unchanged on "
     "Thursday, citing persistent inflationary pressures despite signs of "
     "economic slowdown in the euro zone.",
     "欧洲央行周四决定维持利率不变，理由是尽管欧元区出现经济放缓迹象，但通胀压力依然持续。"),
    ("通用场景", "西班牙语(es→zh)", "es→zh",
     "El cambio climático representa uno de los mayores desafíos de nuestro "
     "tiempo, afectando de manera desproporcionada a las comunidades más vulnerables.",
     "气候变化是我们这个时代面临的最大挑战之一，对最脆弱的群体造成了不成比例的影响。"),
    ("通用场景", "日语(ja→zh)", "ja→zh",
     "私は昨日図書館で借りた本を友達にあげました。",
     "我把我昨天在图书馆借的书给了朋友。"),
]


PROMPTS = {
    "en→zh":  "将以下英文文本翻译成中文：\n{text}\n\n翻译结果：",
    "zh→en":  "Translate the following Chinese text into English:\n{text}\n\nTranslation:",
    "yue→zh": "将以下粤语口语表达转换为标准中文普通话：\n{text}\n\n普通话：",
    "fr→zh":  "将以下法文文本翻译成中文：\n{text}\n\n翻译结果：",
    "de→zh":  "将以下德文文本翻译成中文：\n{text}\n\n翻译结果：",
    "es→zh":  "将以下西班牙文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ru→zh":  "将以下俄文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ar→zh":  "将以下阿拉伯文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ja→zh":  "将以下日文文本翻译成中文：\n{text}\n\n翻译结果：",
    "sw→en":  "Translate the following Swahili text into English:\n{text}\n\nTranslation:",
    "af→en":  "Translate the following Afrikaans text into English:\n{text}\n\nTranslation:",
    "tl→en":  "Translate the following Tagalog text into English:\n{text}\n\nTranslation:",
    "mi→en":  "Translate the following Maori text into English:\n{text}\n\nTranslation:",
}


def load_model(model_path, adapter_path, device):
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  加载基座模型... ({device})")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=dtype, device_map=device if device == "cuda" else "cpu",
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"  加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("  ⚠ 未加载 adapter，使用基座模型")

    model.eval()
    print(f"  完成，耗时 {time.time() - t0:.0f}s")
    return model, tokenizer


def translate(model, tokenizer, text, lang, device):
    template = PROMPTS.get(lang, "将以下文本翻译成中文：\n{text}\n\n翻译结果：")
    prompt = template.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    gen_ids = outputs[0][len(inputs["input_ids"][0]):]
    result = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="翻译模型全面验证")
    parser.add_argument("--model", type=str, required=True, help="基座模型路径")
    parser.add_argument("--adapter", type=str, default="", help="LoRA adapter 路径 (留空则测基座)")
    parser.add_argument("--output", type=str, default="./验证结果.txt", help="结果文件")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0, help="测试条数 (0=全部)")
    args = parser.parse_args()

    cases = TEST_CASES[args.start:]
    if args.count > 0:
        cases = cases[:args.count]

    model_type = "微调后" if args.adapter else "基座（未微调）"
    print("=" * 70)
    print(f"翻译模型验证 — {model_type}")
    print(f"  用例: {len(cases)} 条 | 设备: {args.device}")
    print("=" * 70)

    model, tokenizer = load_model(args.model, args.adapter, args.device)

    print(f"\n运行 {len(cases)} 条测试...\n")

    results = []
    categories = {}
    total_time = 0

    for i, (cat, label, lang, src, expected) in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] {cat} — {label}")
        print(f"  原文: {src[:100]}{'...' if len(src)>100 else ''}")

        translation, elapsed = translate(model, tokenizer, src, lang, args.device)
        total_time += elapsed

        print(f"  翻译: {translation[:120]}{'...' if len(translation)>120 else ''}")
        print(f"  期望: {expected[:120]}{'...' if len(expected)>120 else ''}")
        print(f"  ⏱ {elapsed:.1f}s\n")

        results.append({
            "category": cat, "label": label, "lang": lang,
            "source": src, "expected": expected,
            "translation": translation, "time": elapsed,
        })
        if cat not in categories:
            categories[cat] = {"count": 0, "time": 0}
        categories[cat]["count"] += 1
        categories[cat]["time"] += elapsed

    # 生成报告
    print("=" * 70)
    lines = []
    lines.append("=" * 70)
    lines.append(f"翻译模型验证报告 — {model_type}")
    lines.append(f"总用例: {len(results)} | 总耗时: {total_time:.1f}s | 平均: {total_time/len(results):.1f}s/条")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"{'分类':<14} {'用例':<6} {'耗时':<10} {'平均':<10}")
    lines.append("-" * 40)
    for cat, stats in categories.items():
        avg = stats["time"] / stats["count"]
        lines.append(f"{cat:<14} {stats['count']:<6} {stats['time']:<10.1f}s {avg:<10.1f}s")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r['category']} | {r['label']} | {r['lang']} | {r['time']:.1f}s")
        lines.append(f"原文: {r['source']}")
        lines.append(f"翻译: {r['translation']}")
        lines.append(f"期望: {r['expected']}")
        lines.append("")

    report = "\n".join(lines)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(report.split("\n")[1])
    print(report.split("\n")[2])
    print(f"\n✅ 报告已保存: {args.output}")


if __name__ == "__main__":
    main()

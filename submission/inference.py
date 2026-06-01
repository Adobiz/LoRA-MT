"""
多语种翻译推理脚本
用法:
  python inference.py --text "Hello world" --lang en-zh
  python inference.py --input input.txt --lang en-zh --output result.txt
  python inference.py --interactive
"""
import torch
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "./Hy-MT2-7B"
ADAPTER_PATH = "./final_lora"

PROMPTS = {
    "en-zh":  "将以下英文文本翻译成中文：\n{text}\n\n翻译结果：",
    "zh-en":  "Translate the following Chinese text into English:\n{text}\n\nTranslation:",
    "yue-zh": "将以下粤语口语表达转换为标准中文普通话：\n{text}\n\n普通话：",
    "fr-zh":  "将以下法文文本翻译成中文：\n{text}\n\n翻译结果：",
    "de-zh":  "将以下德文文本翻译成中文：\n{text}\n\n翻译结果：",
    "es-zh":  "将以下西班牙文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ru-zh":  "将以下俄文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ar-zh":  "将以下阿拉伯文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ja-zh":  "将以下日文文本翻译成中文：\n{text}\n\n翻译结果：",
    "ko-zh":  "将以下韩文文本翻译成中文：\n{text}\n\n翻译结果：",
}


def load():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Ready!")
    return model, tokenizer


def translate(model, tokenizer, text, lang):
    template = PROMPTS.get(lang, "将以下文本翻译成中文：\n{text}\n\n翻译结果：")
    prompt = template.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Multilingual Translation Inference")
    parser.add_argument("--text", type=str, help="Source text")
    parser.add_argument("--lang", type=str, default="en-zh")
    parser.add_argument("--input", type=str, help="Input file (one per line)")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--adapter", type=str, default=ADAPTER_PATH)
    args = parser.parse_args()

    global MODEL_PATH, ADAPTER_PATH
    MODEL_PATH = args.model
    ADAPTER_PATH = args.adapter

    model, tokenizer = load()

    if args.interactive:
        print("Interactive mode (type 'q' to quit)")
        while True:
            text = input("\nSource: ").strip()
            if text.lower() == 'q':
                break
            lang = input("Direction (en-zh): ").strip() or "en-zh"
            print(f"Result: {translate(model, tokenizer, text, lang)}")
    elif args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        results = [translate(model, tokenizer, l, args.lang) for l in lines]
        out_file = args.output or args.input.replace(".txt", "_out.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(results))
        print(f"Saved to {out_file}")
    elif args.text:
        result = translate(model, tokenizer, args.text, args.lang)
        print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

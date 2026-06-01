"""
服务器版独立 LoRA 训练脚本
适配 4090D 24GB，支持 Hy-MT2-7B + DoRA
"""
import json
import os
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/autodl-tmp/models/Hy-MT2-7B")
    parser.add_argument("--data_dir", type=str,
                        default="/root/autodl-tmp/LLaMA-Factory/data")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/output_7b")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--use_dora", action="store_true", default=True)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=100000)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    return parser.parse_args()


DATASET_FILES = [
    "translation_专业法律医疗.json",
    "translation_商务商务报告.json",
    "translation_通用新闻类.json",
    "translation_通用日常对话类.json",
    "translation_口语方言口语.json",
]


def load_datasets(data_dir, max_samples):
    """加载所有训练数据集"""
    all_data = []
    for fname in DATASET_FILES:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {fname}: {len(data)} samples")
        all_data.extend(data)

    print(f"  Total: {len(all_data)}")
    if max_samples and len(all_data) > max_samples:
        all_data = all_data[:max_samples]
        print(f"  Limited to: {max_samples}")
    return all_data


def format_example(item):
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output = item.get("output", "")
    prompt = f"{instruction}\n\n输入：{input_text}\n\n输出：" if input_text else f"{instruction}\n\n输出："
    return {"text": prompt + output}


def tokenize_function(examples, tokenizer, cutoff_len):
    model_inputs = tokenizer(
        examples["text"], max_length=cutoff_len, truncation=True, padding=False,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs


def main():
    args = parse_args()
    print("=" * 60)
    print(f"服务器 LoRA 训练")
    print(f"  模型: {args.model_path}")
    print(f"  LoRA rank={args.lora_rank}, DoRA={args.use_dora}")
    print(f"  Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print("=" * 60)

    # ---- 1. GPU ----
    assert torch.cuda.is_available(), "CUDA 不可用！"
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n[GPU] {gpu_name} | {vram:.1f} GB")

    # ---- 2. 数据 ----
    print("\n[数据] 加载...")
    raw_data = load_datasets(args.data_dir, args.max_samples)
    full_dataset = Dataset.from_list(raw_data)
    full_dataset = full_dataset.map(format_example, remove_columns=full_dataset.column_names)
    split = full_dataset.train_test_split(test_size=args.val_size, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"  训练: {len(train_ds)}, 验证: {len(eval_ds)}")

    # ---- 3. 模型 ----
    print("\n[模型] 加载...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.config.use_cache = False
    print(f"  参数: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  显存: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ---- 4. LoRA ----
    print("\n[LoRA] 配置...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        use_dora=args.use_dora,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---- 5. 分词 ----
    print("\n[分词] 处理...")
    train_ds = train_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.cutoff_len),
        batched=True, remove_columns=train_ds.column_names,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_function(x, tokenizer, args.cutoff_len),
        batched=True, remove_columns=eval_ds.column_names,
    )

    # ---- 6. 训练 ----
    total_steps = (len(train_ds) // (args.batch_size * args.grad_accum)) * args.num_epochs
    print(f"\n[训练] 总步数: ~{total_steps}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True),
    )

    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    trainer.train()

    # ---- 保存 ----
    final_path = os.path.join(args.output_dir, "final_lora")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\n✅ 完成！LoRA 模型: {final_path}")
    print(f"  adapter_config.json + adapter_model.safetensors")


if __name__ == "__main__":
    main()

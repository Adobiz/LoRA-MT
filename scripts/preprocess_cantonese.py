#!/usr/bin/env python3
"""粤语数据预处理：占位符保护 → 繁转简 → 还原"""

import opencc
from pathlib import Path

# 白名单：必须保护的粤语特征字（OpenCC 会误伤的 + 安全保护的）
PROTECT_CHARS = {
    "係": "__C00__",  # OpenCC →"系"，粤语义"是"
    "邊": "__C01__",  # OpenCC →"边"，粤语义"哪里"
    "冇": "__C02__",  # 无简体对应但安全起见
    "裏": "__C03__",  # OpenCC →"里"，粤语义"里面"（同"裡"）
    "裡": "__C03__",  # 同"裏"
    "嗰": "__C04__",  # 粤语"那"
    "哋": "__C05__",  # 粤语"们"（我哋=我们）
    "嘅": "__C06__",  # 粤语"的"
    "咗": "__C07__",  # 粤语"了"（完成体）
    "啲": "__C08__",  # 粤语"些"
    "喺": "__C09__",  # 粤语"在"
    "佢": "__C10__",  # 粤语"他"
    "乜": "__C11__",  # 粤语"什么"
    "嘢": "__C12__",  # 粤语"东西/事情"
    "冧": "__C13__",  # 粤语"倒塌/喜欢"
    "瞓": "__C14__",  # 粤语"睡"
    "攰": "__C15__",  # 粤语"累"
    "氹": "__C16__",  # 粤语"哄"
    "脷": "__C17__",  # 粤语"舌头"
    "嚟": "__C18__",  # 粤语"来"
    "冚": "__C19__",  # 粤语"全部/盖"
}

# 反向映射
RESTORE = {v: k for k, v in PROTECT_CHARS.items()}


def protect(text):
    """用占位符替换粤语特征字"""
    result = text
    for char, placeholder in PROTECT_CHARS.items():
        result = result.replace(char, placeholder)
    return result


def restore(text):
    """还原占位符为粤语特征字"""
    result = text
    for placeholder, char in sorted(RESTORE.items(), key=lambda x: -len(x[0])):
        result = result.replace(placeholder, char)
    return result


def main():
    converter = opencc.OpenCC("t2s.json")  # 繁体→简体

    yue_file = Path("D:/huggingface_cache/AI_Tuning/原始语料/口语翻译/方言口语/cantonese_zh.yue")
    backup_file = Path("D:/huggingface_cache/AI_Tuning/原始语料/口语翻译/方言口语/cantonese_zh.yue.orig")

    # 备份原始文件
    if not backup_file.exists():
        import shutil
        shutil.copy2(yue_file, backup_file)
        print(f"已备份: {backup_file}")

    # 读取原始粤语
    lines = yue_file.read_text(encoding="utf-8").splitlines()
    print(f"读取 {len(lines)} 行")

    # 处理
    processed = []
    protect_count = 0
    for line in lines:
        # Step 1: 占位符保护
        protected = protect(line)
        if protected != line:
            protect_count += 1
        # Step 2: 繁转简
        simplified = converter.convert(protected)
        # Step 3: 还原
        restored = restore(simplified)
        processed.append(restored)

    # 写回
    yue_file.write_text("\n".join(processed), encoding="utf-8")
    print(f"已处理 {len(processed)} 行（{protect_count} 行涉及保护字符）")

    # 展示几个例子
    print("\n预处理效果：")
    for i in [0, 1, 2, 10, 100]:
        if i < len(lines):
            print(f"\n  [{i}] 原始: {lines[i][:80]}")
            print(f"  [{i}] 处理后: {processed[i][:80]}")


if __name__ == "__main__":
    main()

import os

# 1. 定义需要清理的关键词列表
# 只要行内包含这些字符串，就会被删除
keywords_to_remove = [
    "- torch==",
    "- torchvision==",
    "- torchaudio==",
    "- triton==",
    "- nvidia-",  # 匹配所有 nvidia 开头的底层库
    "prefix: "    # 删除绝对路径绑定
]

input_file = "environment.yml"
output_file = "environment_clean.yml"

print(f"正在处理 {input_file} ...")
removed_count = 0

try:
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            # 检查当前行是否包含任意一个关键词
            should_remove = False
            for keyword in keywords_to_remove:
                if keyword in line:
                    should_remove = True
                    break
            
            if should_remove:
                removed_count += 1
                # 可以在终端打印出被删除的行，方便核对（可选）
                # print(f"删除: {line.strip()}")
            else:
                f_out.write(line)

    print("-" * 30)
    print(f"✅ 处理完成！共删除了 {removed_count} 行。")
    print(f"📁 新文件已保存为: {output_file}")
    print("-" * 30)
    print("下一步：请使用新文件创建环境：")
    print(f"conda env create -f {output_file}")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {input_file}，请确认文件名是否正确。")
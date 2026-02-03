import sys
import os
sys.path.append(os.path.join(os.getcwd(), "python"))
from llaisys.models.qwen2 import Qwen2

print("🚀 正在加载模型权重...")
try:
    model = Qwen2("/mnt/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B", "cpu")
    model.load_weights()
    print("✅ 权重加载成功！")

    print("🧠 正在尝试生成 1 个 Token...")
    # 151644 是 Qwen 的 <|endoftext|>，872 是 'Hello'
    res = model.generate([151644, 872], max_new_tokens=1)
    print(f"🎉 成功吐出 Token ID: {res}")
except Exception as e:
    print(f"❌ 运行报错: {e}")


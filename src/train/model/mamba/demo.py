from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch
path="/home/xiaomanl/first/mamba-130m-hf" # 修改这里的路径
# 加载模型
tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
model = MambaForCausalLM.from_pretrained(path,local_files_only=True)
input_ids = tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
# 生成输出结果
out = model.generate(input_ids, max_new_tokens=10)
print(tokenizer.batch_decode(out))

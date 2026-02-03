import llaisys
m = llaisys.models.Qwen2('/mnt/workspace/models/DeepSeek-R1-Distill-Qwen-1.5B','cpu')
m.load_weights()
print('calling generate')
m.generate([1,2,3], max_new_tokens=2)
print('done')

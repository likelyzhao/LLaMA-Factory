from modelscope import snapshot_download
import os
model_dir = snapshot_download("qwen/Qwen1.5-7B")
print(model_dir)
exit()

model_dir = snapshot_download('YuanLLM/Yuan2-2B-Mars-hf')
#print("Yuan-2B")
#print(model_dir)
#exit()
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision = "v1.0.5")

model_dir = snapshot_download("baichuan-inc/baichuan-7B", revision = "v1.0.7")
print(model_dir)

model_dir = snapshot_download("langboat/bloom-6b4-zh", revision = "v1.0.0")
print(model_dir)

model_dir = snapshot_download('baicai003/Llama3-Chinese_v2')
print("llma3")
print(model_dir)

model_dir = snapshot_download('qwen/Qwen-7B')
print("qwen7B")
print(model_dir)

model_dir = snapshot_download('01ai/Yi-6B')
print("Yi-6B")
print(model_dir)



model_dir = snapshot_download('AI-ModelScope/OLMo-7B')
print("olmo")
print(model_dir)


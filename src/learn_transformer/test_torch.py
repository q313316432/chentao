# import torch
#
# torch.device("cpu")
#
# a = torch.tensor([[1,2,3],[2,3,4],[4,5,6],[6,7,8]])
#
# a = a.reshape([-1, 4])
#
# print(a.shape)
# print(a)

# 写一下实用huggingface的transformer库执行的文本token化的代码

import torch
from transformers import AutoTokenizer

# 选择一个预训练的分词器，这里以bert-base-uncased为例
model_dir= '/Users/chentao/PycharmProjects/chentao/Qwen2.5-0.5B'

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

# 定义要分词的文本
text = "中国的首都在哪里？"

# 执行分词
ids = tokenizer.encode(text, return_tensors='pt')

# 打印分词结果
print("分词结果：", ids)
print(**ids)

# 如果你想获取分词后的单词列表
token_list = tokenizer.decode(ids[0])
print("分词后的单词列表：", token_list)

# 加载大模型
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
print(model)
# model.save_pretrained("output")
# model.print_trainable_parameters()
# model.eval()
#
# with torch.no_grad():
#     outputs = model.generate(ids)
#
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))







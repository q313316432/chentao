from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

max_seq_length = 50  # 模型处理文本的最大长度

local_model_path = './Qwen2.5-0.5B'

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "ckpts/qwen-1.5b",
    max_seq_length = max_seq_length,
    dtype=None, # 自动检测合适的类型
    load_in_4bit = True,
    # device_map="balanced" # 多卡训练时均衡分布模型权重，默认为sequential
)

# 加载模型 - 修改为适合CPU的加载方式
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float32,  # 在CPU上使用float32
    low_cpu_mem_usage=True,     # 减少CPU内存使用
)

tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# 定义训练数据格式化字符串模板
train_prompt_style = """请遵循指令回答用户问题。
在回答之前，请仔细思考问题，并创建一个逻辑连贯的思考过程，以确保回答准确无误。
### 指令:
你是一位精通八字算命、紫微斗数、风水、易经卦象、塔罗牌占卜、星象、面相手相和运势预测等方面的算命大师。
请回答以下算命问题。
### 问题:
{}
### 回答:
<think>{}</think>
{}
"""

# 加载数据集
dataset = load_dataset("./data/fortune-telling", split="train")

def formatting_data(examples):
    questions = examples["Question"]
    cots = examples["Complex_CoT"]
    responses = examples["Response"]
    texts = []
    for q, c, r in zip(questions, cots, responses):
        text = train_prompt_style.format(q, c, r) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_data, batched=True)
print(dataset[0])

# 添加 LoRA 权重
model = FastLanguageModel.get_peft_model(
    model,
    r = 4, # Rank of the LoRA matrix
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",], # Layers to apply LoRA to
    lora_alpha = 16, # LoRA alpha value
    lora_dropout = 0, # Supports any, but = 0 is optimized，防止过拟合，0 表示不drop任何参数
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

train_args = TrainingArguments(
        per_device_train_batch_size = 2, # 每个GPU上的batch size
        gradient_accumulation_steps = 4, # 梯度累积步数
        warmup_steps = 10,
        # max_steps = 200, # 最大训练步数
        num_train_epochs=3, # 训练轮数 和 max_steps 二选一
        learning_rate = 2e-4, # 学习率，默认值是 2.0e-5
        fp16 = True,
        bf16 = False,
        logging_steps = 2,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    )

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = train_args,
)

# train_stats = trainer.train()

import torch
torch.cuda.is_available()
torch.ze

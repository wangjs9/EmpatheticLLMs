
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 假设你的奖励模型是一个基于Transformers的模型
model_name = "your_reward_model_name"  # 替换为你的奖励模型名称

# 加载预训练的模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 输入数据
user_message = "Hello, how are you?"  # 用户消息
generated_reply = "I'm good, thank you!"  # 模型生成的回复

# 准备输入
inputs = tokenizer(user_message, generated_reply, return_tensors="pt", padding=True, truncation=True)

# 模型推理计算reward score
with torch.no_grad():
    logits = model(**inputs).logits

# 获取奖励分数（假设是logits的输出）
reward_score = logits.item()  # logits的值可能需要进一步处理，取决于模型的具体设计
print(f"Reward score for the reply: {reward_score}")

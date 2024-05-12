from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)

import torch

# model_path = "distilbert-base-uncased"
# 用此数据集训练过 https://huggingface.co/datasets/shawhin/imdb-truncated
model_path = "checkPoints/distilbert-base-uncased-lora-text-classification/checkpoint-2500" # TODO

id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

config = AutoConfig.from_pretrained("distilbert-base-uncased")

# 打印模型参数，内部包含模型的词汇表大小和嵌入维度大小

# 以文本分类模型的方式加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=2, id2label=id2label, label2id=label2id
)

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=False)

if tokenizer.pad_token is None:
    # 添加填充 token，用于填充，使所有的输入序列具有相同的长度
    tokenizer.add_special_tokens({"pad_token": "[PAD]"}) 
    # 调整模型的嵌入大小以匹配新的词汇表长度，因为可能添加了 pad_token
    model.resize_token_embeddings(len(tokenizer)) 


text_list = [
    "It was good.",
    "Not a fan, don't recommend.",
    "Better than the first one.",
    "This is not worth watching even once.",
    "This one is a pass.",
]

print('\n\n\n')

for text in text_list:
    print('\n')
    # 将文本编码为 token ID，并返回 PyTorch 张量
    inputs = tokenizer.encode(text, return_tensors="pt") 
    # 模型输出的未经过概率化的分数，用于分类任务中
    logits = model(inputs).logits
    print('logits is' ,logits)
    # 取出概率最大（分数最高的选项）
    predictions = torch.argmax(logits)
    print('predictions is' ,predictions)
    #将 PyTorch 张量转换为 Python 数值
    id = predictions.tolist()
    print('id is' ,predictions)
    print(text + " - " + id2label[id])

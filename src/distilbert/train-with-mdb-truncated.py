from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from peft import get_peft_model, LoraConfig
import evaluate
import numpy as np
import time

start = time.perf_counter()

dataset = load_dataset("shawhin/imdb-truncated")

np.array(dataset["train"]["label"]).sum() / len(dataset["train"]["label"])

model_checkpoint = "distilbert-base-uncased"

# 定义标签映射
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

# 创建分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# 添加填充 token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# create tokenize function
def tokenize_function(examples):
    text = examples["text"]

    # 分词处理
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text, return_tensors="np", truncation=True, max_length=512
    )

    return tokenized_inputs


# 对数据集进行分词处理
tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

peft_config = LoraConfig(
    task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=["q_lin"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


lr = 1e-3
batch_size = 4
num_epochs = 10

# 训练参数
training_args = TrainingArguments(
    output_dir="checkPoints/" + model_checkpoint, # 训练产生的 checkpoint 存储位置
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# 训练模型
trainer.train()
end = time.perf_counter()
model.to("mps")

print(end - start)

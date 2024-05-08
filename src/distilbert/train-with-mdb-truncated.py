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

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative": 0, "Positive": 1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))


# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["text"]

    # tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text, return_tensors="np", truncation=True, max_length=512
    )

    return tokenized_inputs


# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")


# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

peft_config = LoraConfig(
    task_type="SEQ_CLS", r=4, lora_alpha=32, lora_dropout=0.01, target_modules=["q_lin"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir="checkPoints/" + model_checkpoint,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()
end = time.perf_counter()
model.to("mps")

print(end - start)

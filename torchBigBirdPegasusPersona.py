from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
import torch
import os
# 设置 HF_ENDPOINT 环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 加载模型和分词器
print("加载模型和分词器")
model_name =  "google/bigbird-pegasus-large-arxiv"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BigBirdPegasusForConditionalGeneration.from_pretrained(model_name)
print("模型和分词器加载完成")
# 测试原始模型的输出
input = "Nice to meet you too. What are you interested in?"
print("input question:", input)
input_tokens = tokenizer([input], return_tensors="pt")
output_tokens = model.generate(**input_tokens)
print("output answer:", tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])

print("加载数据集")
# 定义数据集保存路径
dataset_path = "./Persona_valid_preprocessed"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    # 加载预处理后的数据集
    dataset_train = load_from_disk("./Persona_valid_preprocessed")
    dataset_valid = load_from_disk("./Persona_valid_preprocessed")
else:
    dataset = load_dataset("google/Synthetic-Persona-Chat")
    print("dataset finished")
    print("dataset:", dataset)
    print("dataset['train'][0]:", dataset["train"][0])
    dataset_train = dataset["train"]
    dataset_valid = dataset["validation"]
    print("dataset_train:", dataset_train)
    print("dataset_train['Best Generated Conversation'][0]:\n",
          dataset_train["Best Generated Conversation"][0])
    print("dataset_train['user 1 personas'][0]:",
          dataset_train["user 1 personas"][0])
    print("dataset_train['user 2 personas'][0]:",
          dataset_train["user 2 personas"][0])
    print("dataset_train.column_names:",
          dataset_train.column_names)
    # 数据预处理：将对话格式化为上下文-回复对
    def format_dialogue(examples):
        inputs, targets = [], []
        for conversation in examples["Best Generated Conversation"]:
            # 将对话按行拆分
            lines = conversation.split("\n")
            # 将对话拆分为上下文和回复
            # print("lines_range:", len(lines) - 1)
            for i in range(len(lines) - 1):
                context = "\n".join(lines[:i+1])  # 上下文是当前行及之前的所有行
                reply = lines[i+1]  # 下一行是回复
                context = context.replace("User 1: ", "")
                inputs.append(context.strip())
                context = context.replace("User 2: ", "")
                targets.append(reply.strip())
        # print(f"Best Generated Conversation: {len(examples['Best Generated Conversation'])}")
        # print(f"user 1 personas: {len(examples['user 1 personas'])}")
        # print(f"inputs length: {len(inputs)}, targets length: {len(targets)}")
        return {"input": inputs, "target": targets}

    # 应用预处理函数
    dataset_train = dataset_train.map(format_dialogue, batched=True
                                        , remove_columns=["user 1 personas"
                                                            , "user 2 personas"
                                                          , "Best Generated Conversation"])
    dataset_valid = dataset_valid.map(format_dialogue, batched=True
                                        , remove_columns=["user 1 personas"
                                                            , "user 2 personas"
                                                            , "Best Generated Conversation"])
    # 保存预处理后的数据集
    dataset_train.save_to_disk("./Persona_train_preprocessed")
    dataset_valid.save_to_disk("./Persona_valid_preprocessed")
print("tokenizer数据集")
# 定义数据集保存路径
dataset_path = "./PersonaTokenized_train_preprocessed"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    # 加载预处理后的数据集
    dataset_train_tokenized = load_from_disk("./PersonaTokenized_train_preprocessed")
    dataset_valid_tokenized= load_from_disk("./PersonaTokenized_valid_preprocessed")
else:
    # 分词处理
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input"],
            max_length=128,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target"],
                max_length=128,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    dataset_train_tokenized = dataset_train.map(tokenize_function, batched=True)
    dataset_valid_tokenized = dataset_valid.map(tokenize_function, batched=True)
    dataset_train_tokenized = dataset_train_tokenized.filter(lambda example: len(example["input_ids"]) > 0 and len(example["labels"]) > 0)
    dataset_valid_tokenized = dataset_valid_tokenized.filter(lambda example: len(example["input_ids"]) > 0 and len(example["labels"]) > 0)
    # 保存预处理后的数据集
    dataset_train_tokenized.save_to_disk("./PersonaTokenized_train_preprocessed")
    dataset_valid_tokenized.save_to_disk("./PersonaTokenized_valid_preprocessed")
# 计算十分之一的数据量
train_size = len(dataset_train_tokenized)
valid_size = len(dataset_valid_tokenized)
train_subset_size = train_size // 100
valid_subset_size = valid_size // 100
# 使用 select 方法选择前十分之一的数据
dataset_train_tokenized = dataset_train_tokenized.select(range(train_subset_size))
dataset_valid_tokenized = dataset_valid_tokenized.select(range(valid_subset_size))
print("dataset_train_tokenized:",dataset_train_tokenized)
print("dataset_valid_tokenized:",dataset_valid_tokenized)

# 训练参数
TOKENS = 20
EPOCHS = 10
BATCH_SIZE = 4
# 定义训练参数
training_args = TrainingArguments(
    output_dir="./Torch_BigBirdPegasus_persona_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Keep only the last 2 checkpoints
    logging_dir="./torch_logs",            # Directory for logs
    logging_steps=100,               # Log every 100 steps
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    eval_steps=500,                  # Evaluation frequency
    warmup_steps=100,
    learning_rate=5e-5,
    weight_decay=0.01,               # Weight decay
    fp16=True,  # 启用GPU加速（如果有）
)

# 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train_tokenized,
    eval_dataset=dataset_valid_tokenized
)

# 开始训练
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
# 保存模型
model.save_pretrained("./Torch_BigBirdPegasus_persona_finetuned")
tokenizer.save_pretrained("./Torch_BigBirdPegasus_persona_finetuned")
fine_tuned_model = BigBirdPegasusForConditionalGeneration.from_pretrained("./Torch_BigBirdPegasus_persona_finetuned")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./Torch_BigBirdPegasus_persona_finetuned")
# 再次测试对话
print("再次测试对话")
input = "Nice to meet you too. What are you interested in?"
print("input question:", input)
input_tokens = fine_tuned_tokenizer([input], return_tensors="pt")
output_tokens = fine_tuned_model.generate(**input_tokens)
print("output answer:", fine_tuned_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0])
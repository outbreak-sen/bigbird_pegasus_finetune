import os
from datasets import load_dataset, DatasetDict
from transformers import BigBirdPegasusForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch
# 定义数据集保存路径
dataset_path = "./processed_dataset"
# 检查是否存在处理好的数据集
if os.path.exists(dataset_path):
    dataset = DatasetDict.load_from_disk(dataset_path)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
else:
    # 加载和处理数据集
    dataset = load_dataset("databricks/databricks-dolly-15k")
    print(dataset)

    def format_prompt(sample):
        instruction = f"### Instruction\n{sample['instruction']}"
        context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
        response = f"### Answer\n{sample['response']}"
        prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
        sample["prompt"] = prompt
        return sample

    dataset = dataset.map(format_prompt)
    dataset = dataset.remove_columns(['instruction', 'context', 'response', 'category'])
    train_dataset = dataset["train"].select(range(0, 40))
    eval_dataset = dataset["train"].select(range(40, 50))
    # print(train_dataset)
    # print(eval_dataset)
    # print(train_dataset[0])
    # 保存处理好的数据集
    dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    dataset.save_to_disk(dataset_path)

model_name = "google/bigbird-pegasus-large-arxiv"
tokenizer_name = "google/bigbird-roberta-base"
tokenizer_name = "google/bigbird-pegasus-large-arxiv"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token
model = BigBirdPegasusForCausalLM.from_pretrained(model_name)

# 定义自定义 Dataset 类
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["prompt"]
        inputs = self.tokenizer(
            text,
            padding='max_length',  # 填充到固定长度
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"  # 直接返回张量
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),      # shape: (seq_len,)
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": inputs["input_ids"].squeeze(0)          # 用于 CLM 训练
        }

train_dataset = TextDataset(train_dataset, tokenizer)
eval_dataset = TextDataset(eval_dataset, tokenizer)

# 使用 DataCollator 动态填充
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 因为是因果语言模型（Causal LM）
)

TOKENS = 20
EPOCHS = 10
BATCH_SIZE = 4

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./TorchBigbirdFinetune",  # Directory to save the model
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=100,               # Log every 100 steps
    logging_strategy="epoch",
    evaluation_strategy="steps",     # Evaluate every `eval_steps`
    eval_steps=500,                  # Evaluation frequency
    learning_rate=5e-5,              # Learning rate
    weight_decay=0.01,               # Weight decay
   )

# # Step 6: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Use the evaluation dataset
    data_collator=data_collator,
)
trainer.train()
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Step 9: Save the fine-tuned model
model.save_pretrained("./torchModelBigbirdPegasusFinetune")
tokenizer.save_pretrained("./torchTokenizerBigbirdPegasusFinetune")
fine_tuned_model = BigBirdPegasusForCausalLM.from_pretrained("./torchModelBigbirdPegasusFinetune")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./torchTokenizerBigbirdPegasusFinetune")
inputs = fine_tuned_tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = fine_tuned_model(**inputs)
logits = outputs.logits
generated_text = fine_tuned_tokenizer.decode(torch.argmax(logits, dim=-1)[0], skip_special_tokens=True)
print("generated_text", generated_text)



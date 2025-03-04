# !pip install mindnlp
# !pip install mindspore==2.4
# !export LD_PRELOAD=$LD_PRELOAD:/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch.libs/libgomp-74ff64e9.so.1.0.0
# !yum install libsndfile
import os
from mindnlp.transformers import (
    BigBirdPegasusForCausalLM, 
    BigBirdTokenizerFast,
    PegasusTokenizer,
    PreTrainedTokenizerBase)
from datasets import load_dataset, DatasetDict
from mindspore.dataset import GeneratorDataset
from mindnlp.engine import Trainer, TrainingArguments
import mindspore as ms
# 设置运行模式和设备
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend")

# # 定义数据集保存路径
# dataset_path = "./processed_dataset"
# # 检查是否存在处理好的数据集
# if os.path.exists(dataset_path):
#     dataset = DatasetDict.load_from_disk(dataset_path)
#     train_dataset = dataset["train"]
#     eval_dataset = dataset["eval"]
# else:
#     # 加载和处理数据集
#     dataset = load_dataset("databricks/databricks-dolly-15k")
#     print(dataset)

#     def format_prompt(sample):
#         instruction = f"### Instruction\n{sample['instruction']}"
#         context = f"### Context\n{sample['context']}" if len(sample["context"]) > 0 else None
#         response = f"### Answer\n{sample['response']}"
#         prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
#         sample["prompt"] = prompt
#         return sample

#     dataset = dataset.map(format_prompt)
#     dataset = dataset.remove_columns(['instruction', 'context', 'response', 'category'])
#     train_dataset = dataset["train"].select(range(0, 40))
#     eval_dataset = dataset["train"].select(range(40, 50))
#     # print(train_dataset)
#     # print(eval_dataset)
#     # print(train_dataset[0])
#     # 保存处理好的数据集
#     dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
#     dataset.save_to_disk(dataset_path)


# model_name = "google/bigbird-pegasus-large-arxiv"
# tokenizer_name = "google/bigbird-roberta-base"
# tokenizer_name = "google/bigbird-pegasus-large-arxiv"
# # model_name = "./BigBirdPegasus"
# # tokenizer_name = "./BigBirdPegasus"
# # tokenizer = BigBirdTokenizerFast.from_pretrained(tokenizer_name)
# tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)
# tokenizer.pad_token = tokenizer.eos_token # Set padding token这个是什么？
# model = BigBirdPegasusForCausalLM.from_pretrained(model_name)

# class TextDataset:
#     def __init__(self, data):
#         self.data = data
#     # 这里就是个padding和truncation截断的操作
#     def __getitem__(self, index):
#         index = int(index)
#         text = self.data[index]["prompt"]
#         inputs = tokenizer(text, padding='max_length', max_length=256, truncation=True)
#         return (
#             inputs["input_ids"], 
#             inputs["attention_mask"],
#             inputs["input_ids"]  # 添加labels
#         )

#     def __len__(self):
#         return len(self.data)


# train_dataset = GeneratorDataset(
#     TextDataset(train_dataset),
#     column_names=["input_ids", "attention_mask", "labels"],  # 添加labels
#     shuffle=True
# )
# eval_dataset = GeneratorDataset(
#     TextDataset(eval_dataset),
#     column_names=["input_ids", "attention_mask", "labels"],  # 添加labels
#     shuffle=False
# )
# print("train_dataset:", train_dataset)
# print("eval_dataset:", eval_dataset)
# for data in train_dataset.create_dict_iterator():
#     print(data)
#     break

# TOKENS = 20
# EPOCHS = 10
# BATCH_SIZE = 4
# # 定义训练参数
# training_args = TrainingArguments(
#     output_dir='./MindsporeBigBirdFinetune',
#     overwrite_output_dir=True,
#     num_train_epochs=EPOCHS,
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
    
#     save_steps=500,                  # Save checkpoint every 500 steps
#     save_total_limit=2,              # Keep only the last 2 checkpoints
#     logging_dir="./logs",            # Directory for logs
#     logging_steps=100,               # Log every 100 steps
#     logging_strategy="epoch",
#     evaluation_strategy="epoch",
#     eval_steps=500,                  # Evaluation frequency
#     learning_rate=5e-5,
#     weight_decay=0.01,               # Weight decay
# )

# # 创建trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=None
# )
# # 训练模型
# trainer.train()
# eval_results = trainer.evaluate()
# print(f"Evaluation results: {eval_results}")

# # Step 9: Save the fine-tuned model
# model.save_pretrained("./mindNLPModelBigbirdPegasusFinetune")
# tokenizer.save_pretrained("./mindNLPTokenizerBigbirdPegasusFinetune")

# Step 10: Generate text with the fine-tuned model
fine_tuned_model = BigBirdPegasusForCausalLM.from_pretrained("./mindNLPModelBigbirdPegasusFinetune")
fine_tuned_tokenizer = PegasusTokenizer.from_pretrained("./mindNLPTokenizerBigbirdPegasusFinetune")
inputs ="Hello, my dog is cute"
# input_tokens = fine_tuned_tokenizer(input, return_tensors="ms")
# outputs = fine_tuned_model(input_ids=input_tokens["input_ids"],
#                            attention_mask=input_tokens["attention_mask"])
# generated_text = fine_tuned_tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print("generated_text:", generated_text)

inputs = "Hello, my dog is cute"
input_tokens = fine_tuned_tokenizer(inputs, return_tensors="ms")
outputs = fine_tuned_model(**input_tokens)
logits = outputs.logits
# 使用 argmax 获取预测的 token ID
from mindspore import ops
predicted_token_ids = ops.argmax(logits, dim=-1)  # 在最后一个维度（vocab_size）上取 argmax

# 解码生成的文本
generated_text = fine_tuned_tokenizer.decode(predicted_token_ids[0].asnumpy().tolist(), skip_special_tokens=True)
print(generated_text)
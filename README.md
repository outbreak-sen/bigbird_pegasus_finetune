# bigbird_pegasus模型微调对比

这是一个使用bigbird_pegasus模型在Databricks Dolly 15k 数据集上进行微调的测试。

## instruction

BigBird-Pegasus 是基于 BigBird 和 Pegasus 的混合模型，结合了两者的优势，专为处理长文本序列设计。BigBird 是一种基于 Transformer 的模型，通过稀疏注意力机制处理长序列，降低计算复杂度。Pegasus 是专为文本摘要设计的模型，通过自监督预训练任务（GSG）提升摘要生成能力。BigBird-Pegasus 结合了 BigBird 的长序列处理能力和 Pegasus 的摘要生成能力，适用于长文本摘要任务，如学术论文和长文档摘要。

Databricks Dolly 15k 是由 Databricks 发布的高质量指令微调数据集，包含约 15,000 条人工生成的指令-响应对，用于训练和评估对话模型。是专门为NLP模型微调设计的数据集。

## train loss

对比微调训练的loss变化

| epoch | mindnlp+mindspore | transformer+torch（4060） |
| ----- | ----------------- | ------------------------- |
| 1     | 2.9176            | 8.7301                    |
| 2     | 2.79              | 8.1557                    |
| 3     | 2.593             | 7.7516                    |
| 4     | 2.4875            | 7.5017                    |
| 5     | 2.3831            | 7.2614                    |
| 6     | 2.2631            | 7.0559                    |
| 7     | 2.2369            | 6.8405                    |
| 8     | 2.1732            | 6.7297                    |
| 9     | 2.1717            | 6.7136                    |
| 10    | 2.1833            | 6.6279                    |

## eval loss

对比评估得分

| epoch | mindnlp+mindspore  | transformer+torch（4060） |
| ----- | ------------------ | ------------------------- |
| 1     | 2.6390955448150635 | 6.3235931396484375        |
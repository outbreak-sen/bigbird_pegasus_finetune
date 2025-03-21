# bigbird_pegasus的Databricks微调

这是一个使用bigbird_pegasus模型在Databricks Dolly 15k 数据集上进行微调的测试。

## instruction

BigBird-Pegasus 是基于 BigBird 和 Pegasus 的混合模型，结合了两者的优势，专为处理长文本序列设计。BigBird 是一种基于 Transformer 的模型，通过稀疏注意力机制处理长序列，降低计算复杂度。Pegasus 是专为文本摘要设计的模型，通过自监督预训练任务（GSG）提升摘要生成能力。BigBird-Pegasus 结合了 BigBird 的长序列处理能力和 Pegasus 的摘要生成能力，适用于长文本摘要任务，如学术论文和长文档摘要。

## train loss

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

| epoch | mindnlp+mindspore  | transformer+torch（4060） |
| ----- | ------------------ | ------------------------- |
| 1     | 2.6390955448150635 | 6.3235931396484375        |

# bigbird_pegasus的persona微调

实现了bigbird_pegasus模型在google/Synthetic-Persona-Chat数据集上的微调实验。

## train loss

| epoch | mindnlp+mindspore | transformer+torch（3090） |
| ----- | ----------------- | ------------------------- |
| 1     | 0.1826            | 7.6556                    |
| 2     | 0.1614            | 0.5960                    |
| 3     | 0.1435            | 0.4145                    |
| 4     | 0.1398            | 0.3022                    |
| 5     | 0.1344            | 0.2555                    |
| 6     | 0.1263            | 0.2357                    |
| 7     | 0.1200            | 0.2247                    |
| 8     | 0.1147            | 0.2166                    |
| 9     | 0.1105            | 0.2107                    |
| 10    | 0.1082            | 0.2075                    |

## eval loss

| epoch | mindnlp+mindspore | transformer+torch（3090） |
| ----- | ----------------- | ------------------------- |
| 1     | 0.2397            | 0.8738                    |
| 2     | 0.2451            | 0.4804                    |
| 3     | 0.2530            | 0.3490                    |
| 4     | 0.2548            | 0.2861                    |
| 5     | 0.2595            | 0.2669                    |
| 6     | 0.2663            | 0.2612                    |
| 7     | 0.2690            | 0.2545                    |
| 8     | 0.2755            | 0.2526                    |
| 9     | 0.2791            | 0.2519                    |
| 10    | 0.2831            | 0.2510                    |
| 11    | 0.2831            | 0.2510                    |
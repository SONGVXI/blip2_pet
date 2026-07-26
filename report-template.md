# BLIP-2 辅助的图文多模态宠物品种识别实验报告

## 1. 论文信息

### 1.1 BLIP-2

- 论文名称：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- 论文地址：https://arxiv.org/abs/2301.12597
- 官方代码：https://github.com/salesforce/LAVIS/tree/main/projects/blip2

### 1.2 图文融合参考

- 论文名称：Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features
- 论文地址：https://arxiv.org/abs/2001.04732
- 官方代码：https://github.com/AndresPMD/Fine_Grained_Clf

## 2. 任务说明

本实验使用 BLIP-2 为宠物图像生成文本描述，然后将图像特征和文本特征拼接，用于宠物品种分类。

```text
Image -> ResNet-18 -> Image Feature --------┐
                                            ├-> Concatenate -> Classifier
Caption -> Text Encoder -> Text Feature ----┘
```

本实验需要比较：

```text
1. 只使用图像的分类结果；
2. 同时使用图像和 BLIP-2 caption 的分类结果。
```

## 3. 数据集

- 数据集名称：Oxford-IIIT Pet Dataset
- 数据集地址：https://www.robots.ox.ac.uk/~vgg/data/pets/
- 实际使用类别数：10
- 实际使用类别名称：
    "Abyssinian",
    "Bengal",
    "Birman",
    "Persian",
    "Siamese",
    "american_bulldog",
    "american_pit_bull_terrier",
    "english_cocker_spaniel",
    "english_setter",
    "staffordshire_bull_terrier",
- 实际使用图像总数：450
- 训练图像数：300
- 验证图像数：50
- 测试图像数：100


请说明如何选择小型数据子集：

```text
数据抽样使用固定的 random seed，以保证实验结果可以复现。由于 train/validation 和 test 分别来自官方不同的 split，因此不会使用同一张图片。
```

## 4. BLIP-2 Caption 生成

- 使用模型：Salesforce/blip2-opt-2.7b
- 使用 prompt：无，直接使用模型默认的 prompt
- 实际生成 caption 数量：450
- Caption 保存格式：JSON，保存路径为 captions/captions.json


至少展示 3 个 caption 样例：

| 图片编号 | 真实类别 | BLIP-2 Caption |
|:-:|:-:|:-:|
| 1 | english_setter | a white cat |
| 2 | english_cocker_spaniel | a dog laying on a wooden floor |
| 3 | Bengal | a cat laying on the floor in a bathroom |
| 4 | Birman | a white cat with blue eyes |
| 5 | Bengal | two cats on a wheel |


请简要说明 caption 是否能够描述图像中的宠物：

```text
caption可以描述图像中的宠物，包括宠物的品种、颜色、位置、特点等，但是描述caption比较有限。
```

## 5. 数据预处理

### 5.1 图像增强

| 增强方法 | 参数设置 |
|---|---|
| RandomResizedCrop | 224 |
| RandomHorizontalFlip | 0.5 |
| ColorJitter（可选） | 未使用 |
| Normalize | mean=(0.485, 0.456, 0.406)；std=(0.229, 0.224, 0.225) |

### 5.2 文本处理

- 文本编码模型：BLIP-2
- 模型是否冻结：是
- 输入内容：BLIP-2 生成的完整 caption
- 输出特征维度：128
- 文本特征是否提前缓存：是，caption 预先保存到 captions/captions.json


## 6. 模型结构

### 6.1 Image-only 模型

- Image Encoder：ResNet-18
- 是否使用预训练权重：是
- 图像特征维度：512
- 输出类别数：10

模型结构：

```text
Image -> ResNet-18 -> Image Feature -> Linear Classifier
```

### 6.2 Text Encoder

- 实现方式：GRU
- Embedding dimension：128
- Text feature dimension：128

模型结构：

```text
Caption -> Tokenize -> Embedding -> Mean Pooling / GRU -> Text Feature
```

### 6.3 图文拼接模型

- Image feature dimension：512
- Text feature dimension：128
- 拼接后的维度：640
- MLP hidden dimension：256
- 输出类别数：10

```text
Image Feature + Text Feature -> Concatenate -> MLP -> Class Prediction
```

可以粘贴关键代码或伪代码：

```python
# 在这里填写
image_features = self.image_encoder(images)
text_features = self.text_encoder(caption_tokens)
fused_features = torch.cat((image_features, text_features), dim=1)
logits = self.classifier(fused_features)
```

## 7. 训练设置

### 7.1 Image-only

| 配置 | 数值 |
|:-:|:--:|
| epochs | 12 |
| batch size | 32 |
| optimizer | Adam |
| learning rate | 1e-3 |
| loss | CrossEntropyLoss |

### 7.2 Image-Text Fusion

| 配置 | 数值 |
|:-:|:--:|
| epochs | 12 |
| batch size | 32 |
| optimizer | Adam |
| learning rate | 1e-3 |
| loss | CrossEntropyLoss |

## 8. 训练过程

### 8.1 Image-only

| Epoch | Train Loss | Validation Accuracy |
|:--:|:--:|:--:|
|   1   |  1.64439   |        0.36         |
|   2   |  1.222068  |        0.26         |
|   3   |  1.110625  |        0.54         |
|   4   |  1.006686  |        0.34         |
|   5   |  0.871179  |         0.4         |
|   6   |  0.856632  |        0.38         |
|   7   |  0.801999  |         0.4         |
|   8   |  0.714708  |        0.34         |
|   9   |  0.707105  |        0.46         |
|  10   |  0.543928  |         0.5         |
|  11   |  0.515432  |        0.52         |
|  12   |  0.537268  |        0.44         |

请粘贴 loss 曲线、accuracy 曲线或日志截图。

![image_only_loss](C:\Users\Vincent\Downloads\pet_blip2\results\image_only_loss.png)

![image_only_accuracy](C:\Users\Vincent\Downloads\pet_blip2\results\image_only_accuracy.png)

### 8.2 Image-Text Fusion

| Epoch | Train Loss | Validation Accuracy |
|:--:|:--:|:--:|
|   1   |  1.603471  |        0.28         |
|   2   |  1.198911  |         0.4         |
|   3   |  1.11127   |        0.42         |
|   4   |  0.990621  |         0.4         |
|   5   |  0.743455  |        0.28         |
|   6   |  0.685056  |         0.5         |
|   7   |  0.65935   |        0.58         |
|   8   |  0.583592  |        0.66         |
|   9   |  0.509483  |        0.56         |
|  10   |  0.492114  |        0.54         |
|  11   |  0.495667  |        0.62         |
|  12   |  0.392501  |        0.52         |

请粘贴 loss 曲线、accuracy 曲线或日志截图。

![fusion_loss](C:\Users\Vincent\Downloads\pet_blip2\results\fusion_loss.png)

![fusion_accuracy](C:\Users\Vincent\Downloads\pet_blip2\results\fusion_accuracy.png)

请简要描述 loss 是否下降，以及训练是否稳定：

```text
从日志可以看到，两种模型的 Train Loss 都是逐级递减的，但 Validation Accuracy 在不同 epoch 之间存在波动。由于数据量较小，训练 loss 的下降并不代表 validation accuracy 会持续上升，可能存在过拟合和较大的估计波动。
```

## 9. 测试结果

| 模型 | Test Accuracy |
|:-:|:--:|
| Image-only ResNet-18 | 0.43 |
| Image + Caption Fusion | 0.49 |

请分析多模态模型是否优于 image-only 模型：

```text
目前根据数据可以看出来，多模态模型优于 image-only 模型。image-only 的准确率为 43%，Fusion 模型为 49%，高出了 6 个百分点。
```

如果多模态模型没有提升，请分析可能原因：
```text

```

## 10. 预测结果展示

至少展示 5 个测试样例。

| 图片编号 | Caption | 真实类别 | Image-only 预测 | 多模态预测 |
|:-:|:-:|:-:|:-:|:-:|
| 1 | a cat sitting on a window sill | Bengal | Abyssinian | Abyssinian |
| 2 | a dog sitting on a couch | english_setter | english_setter | english_cocker_spaniel |
| 3 | a cat sitting on a chair | Abyssinian | Abyssinian | Abyssinian |
| 4 | a cat laying on a bed | Abyssinian | Persian | Persian |
| 5 | a cat sleeping on a red blanket | Abyssinian | Birman | Birman |

请简单说明文本描述在哪些样例中提供了帮助，在哪些样例中可能产生了干扰：

```text
caption 能够提供动物类型和场景信息，但没有稳定地提供准确的品种标签。文本描述较为通用，未必能够帮助区分相似品种，甚至可能使 prediction 偏向其他类别。而且经过图像增强变换之后的与原图像差距对于模型而言可能过大，容易混淆不同的图像。
```

## 11. 问题与改进

请简要说明：

- 遇到了哪些问题；
- 最终如何解决；
- 图像和 caption 是否出现过对应错误；
- 如果继续改进，可以从哪些方面入手，例如增加数据、调整 prompt、增加 epoch、微调 ResNet 或修改文本编码器。

```text
问题：
    1. BLIP-2 opt-2.7b 的显存占用较高，直接加载会出现 个人GPU 显存不足的问题。
    2. Oxford-IIIT Pet 的图片尺寸不完全一致，需要在 DataLoader 前统一为 224 × 224。
    3. caption 文件中的 image_path 可能是相对路径，而数据读取过程可能产生绝对路径，需要进行路径规范化后再匹配。
    4. 小规模数据集的 validation accuracy 波动较明显。
解决方法：
    - 在 GPU 环境中使用 4-bit quantization、device_map="auto" 和 offload 目录降低 BLIP-2 的显存压力。
    - 在训练和评估中使用 Resize((224, 224))、ToTensor() 和 ImageNet Normalize。
    - 通过路径规范化键将 captions.json 中的 caption 与图片样本匹配。
    - 使用固定 seed、保存 CSV 日志和保存最佳 checkpoint，方便复现和检查。
改进方法：
    - 增加每个类别的图片数量，或使用完整的数据。
    - 增加 `RandomHorizontalFlip`、`RandomResizedCrop` 等 data augmentation。
    - 尝试更强的 text encoder，或比较不同的 `Embedding` 与 `GRU` 配置。
    - 对 ResNet-18 进行部分微调，并补充 per-class precision、recall、F1-score 和 confusion matrix。
```

## 12. AI 对话过程记录

- 录制工具：entire.io
- 对话链接：
- 使用的 AI 模型：GPT 5.6 Luna
- 累计对话时长 / 会话数：总计约4h

简要说明 AI 在哪些环节提供帮助，以及哪些内容由自己检查或修改：

```text
AI 主要协助完成项目目录初始化、数据读取、BLIP-2 caption generation、image-only model、text encoder、Fusion model、training、evaluation、测试脚本。图像增强变换、代码运行结果、训练日志、预测文件和报告内容由自己检查并调试。
```

## 13. Git 提交记录

- 仓库地址：
- 总 commit 数：

粘贴 `git log --oneline` 输出：

```text
（在这里粘贴 git log --oneline）
```


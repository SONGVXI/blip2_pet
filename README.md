# BLIP-2 辅助的图文多模态宠物品种识别复现作业

## 1. 复现论文

本次复现参考论文：

### BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models

- 论文地址：https://arxiv.org/abs/2301.12597
- PDF：https://arxiv.org/pdf/2301.12597
- 官方代码：https://github.com/salesforce/LAVIS/tree/main/projects/blip2
- Hugging Face 文档：https://huggingface.co/docs/transformers/model_doc/blip-2

图文融合部分参考：

### Fine-grained Image Classification and Retrieval by Combining Visual and Locally Pooled Textual Features

- 论文地址：https://arxiv.org/abs/2001.04732
- 官方代码参考：https://github.com/AndresPMD/Fine_Grained_Clf

第二篇论文使用视觉特征与文本特征进行细粒度分类。本次作业只复现其中最基础的思路：分别提取图像特征和文本特征，再将两种特征拼接后完成分类。

本次作业不要求复现 BLIP-2 的大规模预训练，只要求调用已经训练好的 BLIP-2 为图像生成文本描述。之后实现一个轻量化的图文融合分类模型。

考虑到同学需要在一周内使用个人电脑完成，本作业只要求使用约 450 张图像。评价重点是数据读取、caption 生成、模型实现和训练流程是否正确，不以高准确率作为硬性要求。

## 2. 复现目标

实现一个简化的图文多模态分类模型，用于宠物品种识别。

任务形式：

```text
输入：宠物图像，以及 BLIP-2 为该图像生成的文本描述
输出：宠物品种类别
评估：比较只使用图像与同时使用图像和文本时的分类准确率
```

核心目标：

1. 从 Oxford-IIIT Pet 数据集中选取一个约 450 张图像的小子集；
2. 使用冻结的 BLIP-2 为每张图像生成一句英文描述；
3. 使用 ResNet-18 提取图像特征；
4. 使用简单文本编码器提取 caption 特征；
5. 将图像特征和文本特征直接拼接；
6. 训练分类器并报告测试准确率；
7. 与 image-only 模型进行简单对比。

## 3. 数据集要求

使用公开数据集：**Oxford-IIIT Pet Dataset**

- 官方地址：https://www.robots.ox.ac.uk/~vgg/data/pets/
- torchvision 文档：https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html
- PyTorch 可通过 `torchvision.datasets.OxfordIIITPet` 自动下载。

原数据集包含 37 个猫狗品种。本次作业只选择其中 10 个品种，每个品种约 45 张图像，总计约 450 张。

推荐类别：

```text
Abyssinian
Bengal
Birman
Persian
Siamese
american_bulldog
american_pit_bull_terrier
english_cocker_spaniel
english_setter
staffordshire_bull_terrier
```

推荐划分：

```text
训练集：约 300 张
验证集：约 50 张
测试集：约 100 张
```

注意：

- 每个类别在训练集、验证集和测试集中都应有样本；
- 训练集和测试集不能包含同一张图像；
- 数据集不要提交到 Git 仓库；
- 如果电脑性能有限，可以减少到 5 个类别、每类约 50 张图像，但需要在报告中说明。

## 4. Caption 生成要求

使用冻结的 BLIP-2 为图像生成文本描述。

推荐模型：

```text
Salesforce/blip2-opt-2.7b
```

生成流程：

```text
Image
  -> BLIP-2 Processor
  -> Frozen BLIP-2
  -> Caption
  -> 保存到 JSON / txt / CSV 文件
```

要求：

- BLIP-2 只用于推理，不进行训练；
- 每张图像生成一条 caption；
- caption 文件中至少保存图像路径、类别和生成文本；
- 图像与 caption 必须能够正确对应；
- 至少展示 5 个 caption 生成样例；

## 5. 模型结构要求

需要实现 image-only 模型和简单的 image-text fusion 模型。

### 5.1 Image Encoder

可以使用 ImageNet 预训练的 ResNet-18：

```text
Image
  -> ResNet-18
  -> Global Average Pooling
  -> 512-d Image Feature
```

### 5.2 Text Encoder

推荐使用一个简单的文本编码器：

```text
Caption
  -> Tokenize
  -> Embedding
  -> GRU 或平均池化
  -> Text Feature
```

可选实现：

- 最低要求：Embedding 后对所有词向量进行平均池化；
- 推荐要求：Embedding + 单层 GRU。

不要求使用 BERT、Transformer 或复杂预训练文本模型。

### 5.3 拼接融合

最低要求只实现直接拼接：

```text
Image Feature
      +
Text Feature
      -> Concatenate
      -> MLP Classifier
      -> Pet Breed Prediction
```

示例：

```python
image_feature = image_encoder(images)
text_feature = text_encoder(tokens)
fusion_feature = torch.cat([image_feature, text_feature], dim=1)
logits = classifier(fusion_feature)
```

不要求实现 cross-attention、gated fusion、对比学习损失或其他复杂模态交互方法。

### 5.4 推荐整体结构

```text
Image -> ResNet-18 -> Image Feature --------┐
                                            ├-> Concatenate -> MLP -> Class
Caption -> Embedding -> GRU/Mean -> Text Feature ┘
```

## 6. 训练要求

### 6.1 Image-only 模型

```text
epochs: 5-10
batch size: 16 or 32
optimizer: Adam / SGD
learning rate: 1e-3
loss: CrossEntropyLoss
metric: accuracy
```

### 6.2 图文融合模型

```text
epochs: 5-10
batch size: 16 or 32
optimizer: Adam / SGD
learning rate: 1e-3
loss: CrossEntropyLoss
metric: accuracy
```

训练时需要：

- 正确区分 train、validation 和 test；
- 训练时使用 `model.train()`；
- 验证和测试时使用 `model.eval()` 和 `torch.no_grad()`；
- 记录每个 epoch 的 train loss 和 validation accuracy；
- 保存 validation accuracy 最好的模型；
- 最后在测试集上报告 accuracy。

最低结果要求：

- 必须跑通 image-only 和图文融合两套流程；
- 不设置固定准确率门槛；
- 如果多模态结果没有高于 image-only，需要在报告中简单分析可能原因，例如数据量太少、caption 不准确、文本信息有限或模型训练不足。

## 7. 最低完成标准

只要完成以下内容，即认为达到最低复现要求：

1. 能够加载 Oxford-IIIT Pet 小子集；
2. 能够用 BLIP-2 为图像生成 caption，或使用教师 caption 并跑通少量生成样例；
3. 能够根据图像路径读取正确的 caption；
4. 能够搭建并训练 ResNet-18 image-only 分类器；
5. 能够实现简单文本编码器；
6. 能够将图像特征和文本特征拼接后分类；
7. 能够完成训练和测试流程；
8. 能够报告两种模型的测试 accuracy；
9. 能够展示至少 5 个预测样例。

## 8. 最终提交内容

最终提交内容包括：

1. **实现代码**

   需要包含数据读取、caption 生成、模型定义、训练脚本和测试脚本。

2. **实验报告**

   说明参考论文、数据集、caption 生成方法、模型结构、训练设置、实验结果和问题分析。

3. **训练过程记录**

   提供 image-only 和多模态模型的 loss 曲线、accuracy 曲线或关键日志截图。

4. **Caption 结果**

   展示至少 5 个图像及其 BLIP-2 caption。

5. **模型结果**

   至少报告：

   ```text
   使用类别数：
   使用图像数：
   训练 epoch：
   image-only test accuracy：
   image-text fusion test accuracy：
   ```

6. **预测结果展示**

   展示至少 3 张测试图片，给出 caption、真实类别和两个模型的预测类别。

7. **仓库结构**

```text
blip2-pet-fusion/
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── dataset.py
│   ├── generate_captions.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── captions/
│   └── captions.json
├── checkpoints/
├── logs/
├── results/
├── report/
│   └── report.md
```

## 9. 评分建议

总分 100 分：

| 模块 | 分值 | 要求 |
|---|---:|---|
| 数据读取与 Caption | 20 | 正确加载小型数据集，使用 BLIP-2 生成并保存 caption |
| Image-only 模型 | 15 | 正确使用 ResNet-18 并完成训练和测试 |
| 文本编码与图文配对 | 15 | 能编码 caption，并确保图像与文本正确对应 |
| 拼接融合模型 | 15 | 正确拼接图像和文本特征并完成分类 |
| 训练流程 | 15 | 跑通训练、验证和测试，记录 loss 与 accuracy |
| 实验报告 | 15 | 包含 caption、模型结构、结果对比和问题分析 |
| 过程记录 | 5 | 提交 AI 对话记录和 Git 小步提交记录 |

加分项，最多 10 分，总分封顶 100 分：

- 使用 GRU 代替平均词向量；
- 解冻 ResNet-18 最后一层进行微调；
- 对比不同 prompt 生成的 caption；
- 随机打乱 caption，观察分类结果变化；
- 绘制混淆矩阵。

## 10. 过程记录与防作弊要求

为了确认作业是本人逐步完成，而不是一次性生成成品代码，本次复现必须满足以下两条过程性要求。

### 10.1 AI 对话全过程记录

要求使用 entir.io 或同类可分享的对话记录工具，记录与 AI 工具的全部开发对话，并在提交时附上可访问链接。

要求：

```text
- 覆盖范围：读数据、生成 caption、实现模型、训练、调 bug 和写报告
- 不能只记录最后一段成品对话，中间的试错、报错和追问都要保留
- 链接需要可访问，公开或对老师/助教开放
```

提交示例：

```text
AI 对话记录：https://entir.io/s/xxxxxx
使用模型：ChatGPT / Claude / Gemini
累计对话时长：约 3 小时，分 5 次会话
```

### 10.2 Git 小步提交

要求每完成一个小模块提交一次 commit，禁止一次性把整个项目 push 上去。

合格 commit 粒度示例：

```text
feat: 加载 Oxford Pet 小型数据子集
feat: 使用 BLIP-2 生成并保存 caption
feat: 实现 ResNet-18 image-only 模型
feat: 添加简单文本编码器
feat: 实现图像文本拼接融合
feat: 完成训练和验证流程
fix: 修复图像和 caption 对应错误
feat: 添加测试集预测展示
docs: 补充实验结果和问题分析
```

提交时一并附上：

```text
仓库地址：GitHub / Gitee 均可
git log --oneline 的文本输出或截图
```



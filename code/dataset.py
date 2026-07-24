# ********************
# BLIP-2 图文多模态宠物品种识别项目
# Stage 1: 数据集模块 - 模板
# ********************

"""
数据集加载与预处理模块

未来职责：
1. 使用 torchvision 加载 Oxford-IIIT Pet Dataset
   - 自动下载数据集到 data/ 目录
   - 支持 train / test split
2. 图像预处理与数据增强
   - 使用 BLIP-2 对应的 ViT 图像处理器 (Blip2Processor)
   - 训练集数据增强: RandomResizedCrop, RandomHorizontalFlip, ColorJitter
   - 验证集/测试集: Resize + CenterCrop
3. 加载预先生成的 Caption
   - 从 captions/ 目录读取 JSON 格式的 caption 文件
   - 将 caption 文本与对应的图像 ID 关联
4. 构建 PyTorch DataLoader
   - 返回 (image, caption_text, label) 三元组
   - 支持 batch_size 可配置
5. 类别映射
   - 37 个宠物品种类别名称列表
   - class_name -> class_id 映射字典

使用示例:
    from dataset import PetDataset, get_dataloader

    train_loader = get_dataloader(
        split="trainval",
        caption_path="captions/train_captions.json",
        batch_size=32,
        shuffle=True
    )
"""


def main():
    """模板入口，暂不实现具体功能"""
    print("[Stage 1] dataset.py 模块骨架已就位，待后续阶段实现具体功能")


if __name__ == "__main__":
    main()
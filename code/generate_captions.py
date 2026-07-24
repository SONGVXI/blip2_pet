# ********************
# BLIP-2 图文多模态宠物品种识别项目
# Stage 1: Caption 生成模块 - 模板
# ********************

"""
BLIP-2 Caption 生成脚本

未来职责：
1. 加载预训练 BLIP-2 模型
   - 使用 transformers 库加载 Salesforce/blip2-opt-2.7b
   - 模型加载到 GPU (如果可用)
2. 遍历 Oxford-IIIT Pet Dataset 所有图像
   - 读取 train/val/test 三个 split 的图像
   - 每张图像调用 BLIP-2 生成 caption
3. Caption 生成策略
   - 使用 beam search 或 nucleus sampling 解码
   - 可选 prompt 引导 (如 "a photo of a")
4. 生成结果保存
   - 保存为 JSON 格式到 captions/ 目录
   - 每张图像对应一条记录: {image_id, caption, split}
5. 进度显示
   - 使用 tqdm 显示生成进度

使用示例:
    python generate_captions.py --split trainval --output captions/train_captions.json
"""


def main():
    """模板入口，暂不实现具体功能"""
    print("[Stage 1] generate_captions.py 模块骨架已就位，待后续阶段实现具体功能")


if __name__ == "__main__":
    main()
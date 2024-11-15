import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import os
from skimage import measure
import pandas as pd

# 读取txt文件并解析边界框
def read_boxes_from_txt(file_path):
    boxes = []
    with open(file_path, 'r') as file:
        for line in file:
            coords = list(map(int, line.strip().split()))
            if len(coords) == 4:
                boxes.append(coords)
    return boxes

# 配置模型类型和检查点路径
sam_checkpoint = "./sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载 SAM 模型
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 创建预测器
predictor = SamPredictor(sam)

# 定义输入和输出目录
input_dir = './test_image/C_aug' # 图像路径
output_dir = './results/C_Nihe' # 检测结果路径
os.makedirs(output_dir, exist_ok=True)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 设置长轴长度的阈值
major_axis_threshold = 200 # 根据需要调整这个值

# 处理每个图像文件
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, image_file)
    image_name = os.path.splitext(image_file)[0]
    
    # 读取并预处理图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像文件 {image_path}，跳过")
        continue
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 设置图像到预测器
    predictor.set_image(image)

    # 读取并转换边界框
    boxes_path = os.path.join(output_dir, f'{image_name}_detected_boxes.txt')
    if not os.path.exists(boxes_path):
        print(f"边界框文件 {boxes_path} 不存在，跳过 {image_file}")
        continue

    custom_boxes = read_boxes_from_txt(boxes_path)
    input_boxes = torch.tensor(custom_boxes, device=predictor.device)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    print(f"开始分割图像 {image_file}")

    # 分批处理边界框，避免一次性处理所有边界框
    batch_size = 50  # 调整批处理大小以适应可用内存
    masks = []
    for i in tqdm(range(0, len(transformed_boxes), batch_size), desc="Processing batches"):
        batch_boxes = transformed_boxes[i:i + batch_size]
        batch_masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=batch_boxes,
            multimask_output=False,
        )
        masks.extend(batch_masks)

    print(f"分割完成，开始保存灰度掩码 {image_file}")

    # 保存灰度掩码到PNG文件
    mask_all = np.zeros_like(masks[0].cpu().numpy(), dtype=np.uint8)
    for idx, mask in enumerate(masks):
        unique_color = idx + 1  # 为每个掩码分配一个唯一的颜色索引
        mask_all = np.maximum(mask_all, mask.cpu().numpy().astype(np.uint8) * unique_color)

    # 移除多余的维度
    mask_all = np.squeeze(mask_all)

    # 确保掩码和图像形状一致
    if mask_all.shape != image.shape[:2]:
        mask_all_resized = cv2.resize(mask_all, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask_all_resized = mask_all
    
    # 绘制椭圆到原图像上
    overlay_image = image.copy()
    for region_label in np.unique(mask_all_resized):
        if region_label == 0:
            continue  # 跳过背景
        
        region_mask = (mask_all_resized == region_label).astype(np.uint8)
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if contour.shape[0] >= 5:  # 椭圆拟合需要至少5个点
                ellipse = cv2.fitEllipse(contour)
                major_axis_length = max(ellipse[1])  # 获取长轴长度
                if major_axis_length <= major_axis_threshold:
                    # 仅绘制宽度和高度为正值的椭圆
                    if ellipse[1][0] > 0 and ellipse[1][1] > 0:
                        cv2.ellipse(overlay_image, ellipse, (0, 255, 0), 2)  # 绿色绘制椭圆
    
    # 保存叠加结果图像
    overlay_path = os.path.join(output_dir, f'{image_name}_ellipse_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

    print(f"处理完成 {image_file}")

print("所有图像处理完成")

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import os
from skimage.color import label2rgb
from skimage.measure import regionprops_table
import pandas as pd
from skimage import measure

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
input_dir = './test_image/2d' # 图像路径
output_dir = './results/2d' # 检测结果路径
os.makedirs(output_dir, exist_ok=True)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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
    
    mask_path = os.path.join(output_dir, f'{image_name}_mask.png')
    cv2.imwrite(mask_path, mask_all_resized * 255)

    # 使用skimage的label2rgb函数将标签颜色叠加到原图
    labels = mask_all_resized
    # 生成独特的颜色
    num_labels = labels.max() + 1
    colors = np.random.rand(num_labels, 3)
    label_image = label2rgb(labels, colors=colors, bg_label=0, alpha=1.0)

    # 叠加掩码到原图像上
    overlay_image = cv2.addWeighted(image, 0.7, (label_image * 255).astype(np.uint8), 0.5, 0)

    # 绘制轮廓到原图像上
    contours = measure.find_contours(mask_all_resized, 0.5)
    for contour in contours:
        contour = np.flip(contour, axis=1).astype(int)
        cv2.drawContours(overlay_image, [contour], -1, (0, 0, 255), 1)  # 用红色绘制轮廓

    # 保存叠加结果图像
    overlay_path = os.path.join(output_dir, f'{image_name}_overlay.png')
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))

    print(f"处理完成 {image_file}")

    # 使用 regionprops_table 获取每个掩码的属性
    props = regionprops_table(mask_all_resized, properties=('label', 'area', 'equivalent_diameter', 'feret_diameter_max', 'major_axis_length', 'minor_axis_length', 'perimeter'))
    n_of_units = 590 # centimeters in the case of 'IMG_5208_image.png'
    units_per_pixel = n_of_units / 1920
    df = pd.DataFrame(props)
    df['area'] = (df['area'] * units_per_pixel**2).round(3)
    df['equivalent_diameter'] = (df['equivalent_diameter'].values * units_per_pixel).round(3)
    df['feret_diameter_max'] = (df['feret_diameter_max'].values * units_per_pixel).round(3)
    df['major_axis_length'] = (df['major_axis_length'].values * units_per_pixel).round(3)
    df['minor_axis_length'] = (df['minor_axis_length'].values * units_per_pixel).round(3)
    df['perimeter'] = (df['perimeter'].values * units_per_pixel).round(3)
    props_path = os.path.join(output_dir, f'{image_name}_props.csv')
    df.to_csv(props_path, index=False)
    print(f"属性提取完成，保存到 {props_path}")

print("所有图像处理完成")

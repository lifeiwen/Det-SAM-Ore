import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import trange, tqdm
import os

def detect_image_tiles(big_im, model, tile_height, tile_width):
    # 获取图像的尺寸
    height, width, _ = big_im.shape

    # 创建空的图像用于合并检测结果
    big_im_result = np.copy(big_im)

    # 滑动窗口的步长
    stride_height = tile_height  # 设置为与tile_height相同
    stride_width = tile_width    # 设置为与tile_width相同

    # 存储所有检测框坐标的列表
    all_boxes = []

    # 迭代处理图像的每个块
    for y in trange(0, height, stride_height, desc="Processing rows"):
        for x in range(0, width, stride_width):
            # 切片图像块
            im_tile = big_im[y:y+tile_height, x:x+tile_width]
            
            # 检查图像块的尺寸是否足够大
            if im_tile.shape[0] < tile_height or im_tile.shape[1] < tile_width:
                pad_bottom = max(0, tile_height - im_tile.shape[0])
                pad_right = max(0, tile_width - im_tile.shape[1])
                im_tile = cv2.copyMakeBorder(im_tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            # 对图像块进行检测
            #results = model(im_tile, max_det=2000)
            results = model(im_tile, max_det=2000, conf=0.5)  # 设置置信度阈值为0.5
            
            # 获取检测框坐标并转换为相对于整个大图像的坐标
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()  # 提取检测框坐标
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y
                    all_boxes.append([x1, y1, x2, y2])
                    color = (0, 255, 0)  # 设置颜色
                    cv2.rectangle(big_im_result, (x1, y1), (x2, y2), color, 2)
    
    return big_im_result, all_boxes

# 加载训练好的模型
#model = YOLO("./runs0704/train/yolov8x/weights/best.pt")
model = YOLO("./runs/train/yolov8x2/weights/best.pt")
#model = YOLO("./runs/train/yolov8x-MultiSEAMHead/weights/best.pt")
# 定义输入和输出目录
input_dir = './test_image/C_aug/'
output_dir = './results/C_Nihe/'
os.makedirs(output_dir, exist_ok=True)

# 获取文件夹中所有图像文件
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]

# 总计数器
total_boxes_count = 0

# 处理每个图像文件
for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, image_file)
    big_image = cv2.imread(image_path)
    image_name = os.path.splitext(image_file)[0]

    # 检测并合并结果
    detected_image, all_boxes = detect_image_tiles(big_image, model, 1000, 1000)

    # 保存结果图像
    output_image_path = os.path.join(output_dir, f'{image_name}_detected.jpg')
    cv2.imwrite(output_image_path, detected_image)

    # 将检测框坐标写入txt文件
    output_txt_path = os.path.join(output_dir, f'{image_name}_detected_boxes.txt')
    with open(output_txt_path, 'w') as f:
        for box in all_boxes:
            x1, y1, x2, y2 = box
            f.write(f'{x1} {y1} {x2} {y2}\n')
    
    # 累加当前图像的检测框数量
    total_boxes_count += len(all_boxes)
    print(f'{image_file} 检测到的坐标数量: {len(all_boxes)}')
    print(f'检测完成并保存结果图像为 {output_image_path} 和坐标为 {output_txt_path}')

# 输出总共检测到的坐标数量
print(f'总共检测到的坐标数量: {total_boxes_count}')

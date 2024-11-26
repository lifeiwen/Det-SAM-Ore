import numpy as np
from ultralytics import YOLO
import cv2
from tqdm import trange, tqdm
import oshttps://github.com/lifeiwen/Det-SAM-Ore/blob/main/Ore_Detect_Predict.py

def detect_image_tiles(big_im, model, tile_height, tile_width):
 
    height, width, _ = big_im.shape

  
    big_im_result = np.copy(big_im)

 
    stride_height = tile_height  
    stride_width = tile_width    


    all_boxes = []


    for y in trange(0, height, stride_height, desc="Processing rows"):
        for x in range(0, width, stride_width):
         
            im_tile = big_im[y:y+tile_height, x:x+tile_width]
            
     
            if im_tile.shape[0] < tile_height or im_tile.shape[1] < tile_width:
                pad_bottom = max(0, tile_height - im_tile.shape[0])
                pad_right = max(0, tile_width - im_tile.shape[1])
                im_tile = cv2.copyMakeBorder(im_tile, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            
            #results = model(im_tile, max_det=2000)
            results = model(im_tile, max_det=2000, conf=0.5) 
            
        
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy() 
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1 += x
                    y1 += y
                    x2 += x
                    y2 += y
                    all_boxes.append([x1, y1, x2, y2])
                    color = (0, 255, 0)  
                    cv2.rectangle(big_im_result, (x1, y1), (x2, y2), color, 2)
    
    return big_im_result, all_boxes


#model = YOLO("./runs0704/train/yolov8x/weights/best.pt")
model = YOLO("./runs/train/yolov8x2/weights/best.pt")
#model = YOLO("./runs/train/yolov8x-MultiSEAMHead/weights/best.pt")

input_dir = './test_image/C_aug/'
output_dir = './results/C_Nihe/'
os.makedirs(output_dir, exist_ok=True)


image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]


total_boxes_count = 0


for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, image_file)
    big_image = cv2.imread(image_path)
    image_name = os.path.splitext(image_file)[0]


    detected_image, all_boxes = detect_image_tiles(big_image, model, 1000, 1000)


    output_image_path = os.path.join(output_dir, f'{image_name}_detected.jpg')
    cv2.imwrite(output_image_path, detected_image)


    output_txt_path = os.path.join(output_dir, f'{image_name}_detected_boxes.txt')
    with open(output_txt_path, 'w') as f:
        for box in all_boxes:
            x1, y1, x2, y2 = box
            f.write(f'{x1} {y1} {x2} {y2}\n')
    

    total_boxes_count += len(all_boxes)
    print(f'{image_file} 检测到的坐标数量: {len(all_boxes)}')
    print(f'检测完成并保存结果图像为 {output_image_path} 和坐标为 {output_txt_path}')


print(f'总共检测到的坐标数量: {total_boxes_count}')

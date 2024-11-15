import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch
import gc

def train_model():
    model = YOLO('ultralytics/cfg/models/v8/Ore-Detect.yaml')
    model.load('yolov8x.pt')  # loading pretrain weights
    model.train(
        data='./dataset/yolo_data/data.yaml',
        cache=False,
        imgsz=640,
        epochs=100,
        batch=4,
        close_mosaic=0,
        workers=4,
        device='0',
        optimizer='SGD',  # using SGD
        # patience=0,  # close earlystop
        # resume='',  # last.pt path
        # amp=False,  # close amp
        # fraction=0.2,
        multi_scale = True,
        project='runs0704/train',
        name='Ore-Detect',
    )

if __name__ == '__main__':
    train_model()

    # 显式释放资源
    torch.cuda.empty_cache()
    gc.collect()

import cv2

import pycuda.autoinit
from src.yolo import YOLO
import tensorrt as trt

if __name__ == "__main__":
    yolo = YOLO(trt.Logger())
    img = cv2.imread("models/sample_table.png")
    print(yolo(img))

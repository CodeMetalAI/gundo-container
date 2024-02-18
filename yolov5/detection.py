import torch
import cv2
from time import time

def get_model(name = "yolov5n"):
    return torch.hub.load("ultralytics/yolov5", name)  # or yolov5n - yolov5x6, custom

def detect(img, model):
    results = model(img)
    boxes = results.crop()
    return boxes

if __name__ == "__main__":
    # Model
    model = torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n - yolov5x6, custom

    # Images
    img = "zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    im = cv2.imread(img)

    # Inference
    t0 = time()
    results = model(img)
    print(time() - t0)
    boxes = results.crop()

    for box in boxes:
        bbox = box["box"]
        conf = box["conf"]
        label = box["label"].split(" ")[0]

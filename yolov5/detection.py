import torch
import cv2
from time import time

def get_model(name = "custom"):
    if name == "custom":
        return torch.hub.load('ultralytics/yolov5', 'custom', './yolov5/last.pt')
    return torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n - yolov5x6, custom

def detect(img, model):
    results = model(img)
    boxes = results.crop()
    return boxes

if __name__ == "__main__":
    import sys
    import os

    # Model
    #model = torch.hub.load("ultralytics/yolov5", "yolov5n")  # or yolov5n - yolov5x6, custom

    #model = torch.load("last.pt")
    #model = torch.hub.load("/Users/tzofi/Documents/Code/BounceFlashLidar/Code/gundo-container/yolov5", "last", source="local")

    model = torch.hub.load('ultralytics/yolov5', 'custom', './last.pt')

    # Images
    #img = "zidane.jpg"  # or file, Path, PIL, OpenCV, numpy, list
    #im = cv2.imread(img)

    for im in os.listdir(sys.argv[1]):
        im = cv2.imread(os.path.join(sys.argv[1], im))
        t0 = time()
        results = model(im)
        print(time() - t0)
        results.save()

    #boxes = results.crop()

    #for box in boxes:
    #    bbox = box["box"]
    #    conf = box["conf"]
    #    label = box["label"].split(" ")[0]

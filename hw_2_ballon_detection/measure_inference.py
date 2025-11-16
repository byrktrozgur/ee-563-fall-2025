import os
import glob
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.transforms import functional as F

from maskrcnn_train_model import get_instance_segmentation_model

IMAGE_DIR = "balloon_dataset/images/val"
YOLO_WEIGHTS = "runs/segment/train/weights/best.pt"
MASKRCNN_WEIGHTS = "maskrcnn_balloon.pth"
NUM_CLASSES = 2

def benchmark_yolo(device_str="cuda", num_warmup=5, num_iters=20):
    device = device_str
    model = YOLO(YOLO_WEIGHTS)

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    img = cv2.imread(image_paths[0])
    if img is None:
        print("Failed to read image")
        return

    # warmup
    for _ in range(num_warmup):
        _ = model(img, device=device, verbose=False)

    # measure
    times = []
    for _ in range(num_iters):
        t0 = time.perf_counter()
        _ = model(img, device=device, verbose=False)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    print(f"YOLO on {device_str}: {avg_ms:.2f} ms / image")


def benchmark_maskrcnn(device_str="cuda", num_warmup=5, num_iters=20):
    device = torch.device(device_str)
    model = get_instance_segmentation_model(NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(MASKRCNN_WEIGHTS, map_location=device))
    model.to(device)
    model.eval()

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    img_bgr = cv2.imread(image_paths[0])
    if img_bgr is None:
        print("Failed to read image")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(img_tensor)

    # measure
    times = []
    with torch.no_grad():
        for _ in range(num_iters):
            t0 = time.perf_counter()
            _ = model(img_tensor)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_ms = 1000.0 * sum(times) / len(times)
    print(f"Mask R-CNN on {device_str}: {avg_ms:.2f} ms / image")


def main():
    # GPU (if available)
    if torch.cuda.is_available():
        benchmark_yolo("cuda")
        benchmark_maskrcnn("cuda")
    else:
        print("CUDA not available, skipping GPU benchmarks.")

    # CPU
    benchmark_yolo("cpu")
    benchmark_maskrcnn("cpu")


if __name__ == "__main__":
    main()

import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO
from torchvision.transforms import functional as F

from maskrcnn_train_model import get_instance_segmentation_model

# ============================================================
# 1. Build evaluation dataset (ground-truth) from YOLO labels
# ============================================================
# Use your actual paths here (typically val split)
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path="balloon_dataset/images/val",
    annotations_directory_path="balloon_dataset/labels/val",
    data_yaml_path="balloon_dataset\\balloon.yml",   # or your data yaml if needed
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. YOLO callback + mAP
# ============================================================
yolo_model = YOLO("runs/detect/train/weights/best.pt")  # or your path

def yolo_callback(image: np.ndarray) -> sv.Detections:
    # image is a numpy array (H,W,3) in BGR or RGB; YOLO handles np.ndarray
    result = yolo_model(image)[0]
    # supervision has a direct converter for Ultralytics
    return sv.Detections.from_ultralytics(result)

yolo_map = sv.MeanAveragePrecision.benchmark(
    dataset=dataset,
    callback=yolo_callback,
)
print(f"YOLO mAP@0.5:      {yolo_map.map50:.4f}")

# ============================================================
# 3. Mask R-CNN callback + mAP
# ============================================================
NUM_CLASSES = 2
mask_model = get_instance_segmentation_model(NUM_CLASSES, pretrained=False)
mask_model.load_state_dict(torch.load("maskrcnn_balloon.pth", map_location=device))
mask_model.to(device)
mask_model.eval()

def maskrcnn_callback(image: np.ndarray) -> sv.Detections:
    # image is BGR from supervision/YOLO style; flip to RGB and copy to make strides positive
    image_rgb = image[..., ::-1].copy()  # <<< THIS .copy() FIXES THE ERROR

    img_tensor = F.to_tensor(image_rgb).to(device)

    with torch.no_grad():
        output = mask_model([img_tensor])[0]

    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()

    # keep almost everything
    keep = scores > 0.001
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Mask R-CNN: 0 = background, 1 = balloon
    # YOLO dataset: 0 = balloon
    labels = labels - 1  # 1 -> 0

    return sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=labels,
    )


mask_map = sv.MeanAveragePrecision.benchmark(
    dataset=dataset,
    callback=maskrcnn_callback,
)

print(f"Mask R-CNN mAP@0.5:      {mask_map.map50:.4f}")
import os
import glob

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# ================== COMMON CONFIG ==================
IMAGE_DIR = r"balloon_dataset\\images\\test"          # folder with input images

# YOLO config
YOLO_WEIGHTS_PATH = r"runs\\detect\\train\\weights\\best.pt"
YOLO_OUTPUT_DIR   = r"results\\YOLO"

# Mask R-CNN config
MASK_WEIGHTS_PATH = r"maskrcnn_balloon.pth"
MASK_OUTPUT_DIR   = r"results\\MaskRCNN"
MASK_SCORE_THRESH = 0.5
MASK_NUM_CLASSES  = 2               # background + balloon
# ===================================================


# ================== YOLO PART ==================
def compute_bbox_area_percent(result, w_img, h_img):
    """
    Compute estimated balloon area (%) using HALF of the bounding box area.

    Formula:
        bbox_area      = (x2 - x1) * (y2 - y1)
        balloon_area   = 0.5 * bbox_area
        area_percent   = 100 * balloon_area / (w_img * h_img)
    Returns:
      - area_percent (float) or None if no detection
      - (x1, y1, x2, y2) bounding box (ints) or None
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None

    boxes = result.boxes

    # choose highest confidence detection
    confs = boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))

    # xyxy coordinates
    xyxy = boxes.xyxy[idx].cpu().numpy()
    x1, y1, x2, y2 = map(int, xyxy)

    # bounding box area
    bbox_area = (x2 - x1) * (y2 - y1)

    # estimated balloon area = 0.5 * bbox_area
    balloon_area = 0.5 * bbox_area

    total_area = w_img * h_img
    area_percent = 100.0 * balloon_area / total_area

    return area_percent, (x1, y1, x2, y2)


def draw_bbox(image_bgr, bbox):
    """
    Draw a red bounding box on the image and return the annotated image.
    """
    vis = image_bgr.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red bbox
    return vis


def run_yolo():
    if not os.path.isfile(YOLO_WEIGHTS_PATH):
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS_PATH}")
    if not os.path.isdir(IMAGE_DIR):
        raise NotADirectoryError(f"Image directory not found: {IMAGE_DIR}")

    os.makedirs(YOLO_OUTPUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[YOLO] Using device: {device}")

    model = YOLO(YOLO_WEIGHTS_PATH)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    print(f"[YOLO] Found {len(image_paths)} images.")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n[YOLO] Processing: {img_name}")

        img = cv2.imread(img_path)
        if img is None:
            print("  Could not read image, skipping:", img_path)
            continue

        h_img, w_img = img.shape[:2]

        results = model(img, device=device, verbose=False)
        result = results[0]

        area_percent, bbox = compute_bbox_area_percent(result, w_img, h_img)
        if area_percent is None or bbox is None:
            print("  No balloon detected.")
            out_path = os.path.join(
                YOLO_OUTPUT_DIR,
                os.path.splitext(img_name)[0] + "_annotated.jpg",
            )
            cv2.imwrite(out_path, img)
            print("  Saved (no detection) image to:", out_path)
            continue

        print(f"  Estimated balloon area: {area_percent:.2f}% of the image")

        annotated = draw_bbox(img, bbox)

        text = f"Area: {area_percent:.2f}%"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = os.path.join(
            YOLO_OUTPUT_DIR,
            os.path.splitext(img_name)[0] + "_annotated.jpg",
        )
        cv2.imwrite(out_path, annotated)
        print("  Saved annotated image to:", out_path)


# ================== MASK R-CNN PART ==================
def get_mask_model(num_classes, weights_path, device):
    model = maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Mask R-CNN weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def compute_mask_area_percent(output, w_img, h_img, score_thresh=0.5):
    """
    Compute balloon area (%) from highest-score mask prediction.
    Returns:
      - area_percent (float) or None
      - bbox (x1,y1,x2,y2) or None
      - mask (H,W) uint8 0/1 or None
    """
    if len(output["scores"]) == 0:
        return None, None, None

    scores = output["scores"].detach().cpu().numpy()
    keep_idx = np.where(scores >= score_thresh)[0]
    if len(keep_idx) == 0:
        return None, None, None

    best = keep_idx[np.argmax(scores[keep_idx])]

    boxes = output["boxes"][best].detach().cpu().numpy()
    x1, y1, x2, y2 = boxes.astype(int)
    bbox = (x1, y1, x2, y2)

    mask_prob = output["masks"][best, 0].detach().cpu().numpy()
    mask = (mask_prob >= 0.5).astype(np.uint8)

    mask_area = float(mask.sum())
    total_area = float(w_img * h_img)
    area_percent = 100.0 * mask_area / total_area

    return area_percent, bbox, mask


def overlay_mask(image_bgr, mask, color=(0, 0, 255), alpha=0.4):
    """
    Overlay a single-channel mask onto an image.
    """
    overlay = image_bgr.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = color

    return cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)


def run_maskrcnn():
    if not os.path.isdir(IMAGE_DIR):
        raise NotADirectoryError(f"Image directory not found: {IMAGE_DIR}")

    os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Mask R-CNN] Using device: {device}")

    model = get_mask_model(MASK_NUM_CLASSES, MASK_WEIGHTS_PATH, device)

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    if not image_paths:
        print("No images found in:", IMAGE_DIR)
        return

    print(f"[Mask R-CNN] Found {len(image_paths)} images.")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        print(f"\n[Mask R-CNN] Processing: {img_name}")

        img_pil = Image.open(img_path).convert("RGB")
        w_img, h_img = img_pil.size
        img_tensor = F.to_tensor(img_pil).to(device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)

        output = outputs[0]
        area_percent, bbox, mask = compute_mask_area_percent(
            output, w_img, h_img, MASK_SCORE_THRESH
        )

        img_cv = cv2.imread(img_path)
        if img_cv is None:
            print("  Could not read image with OpenCV, skipping:", img_path)
            continue

        if area_percent is None or bbox is None or mask is None:
            print("  No balloon detected above threshold.")
            out_path = os.path.join(
                MASK_OUTPUT_DIR,
                os.path.splitext(img_name)[0] + "_annotated.jpg",
            )
            cv2.imwrite(out_path, img_cv)
            print("  Saved (no detection) image to:", out_path)
            continue

        print(f"  Estimated balloon area (mask): {area_percent:.2f}% of the image")

        annotated = overlay_mask(img_cv, mask, color=(0, 0, 255), alpha=0.4)

        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"Area: {area_percent:.2f}%"
        cv2.putText(
            annotated,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = os.path.join(
            MASK_OUTPUT_DIR,
            os.path.splitext(img_name)[0] + "_annotated.jpg",
        )
        cv2.imwrite(out_path, annotated)
        print("  Saved annotated image to:", out_path)


# ================== ENTRY POINT ==================
def main():
    print("Choose model for balloon area computation:")
    print("  1) YOLO")
    print("  2) Mask R-CNN")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        run_yolo()
    elif choice == "2":
        run_maskrcnn()
    else:
        print("Invalid choice. Please run again and choose 1 or 2.")


if __name__ == "__main__":
    main()

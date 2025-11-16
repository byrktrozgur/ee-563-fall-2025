# maskrcnn_train_balloon.py

import os
import json
import time
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ================== CONFIG ==================
DATA_DIR = "balloon_dataset"

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "images", "train")
VAL_IMG_DIR   = os.path.join(DATA_DIR, "images", "val")

TRAIN_COCO = os.path.join(DATA_DIR, "annotations_coco_train.json")
VAL_COCO   = os.path.join(DATA_DIR, "annotations_coco_val.json")

NUM_CLASSES = 2  # 1 class (balloon) + background
BATCH_SIZE  = 2
NUM_EPOCHS  = 20
LR          = 0.005
MOMENTUM    = 0.9
WEIGHT_DECAY = 0.0005

OUTPUT_WEIGHTS = "maskrcnn_balloon.pth"
NUM_WORKERS    = 0
# ============================================


class BalloonCocoDataset(Dataset):
    def __init__(self, img_dir, coco_json_path, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        ann_by_img = defaultdict(list)
        for ann in self.annotations:
            ann_by_img[ann["image_id"]].append(ann)
        self.ann_by_img = ann_by_img

        self.id_to_img = {im["id"]: im for im in self.images}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_path = os.path.join(self.img_dir, file_name)

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        anns = self.ann_by_img.get(img_id, [])

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann.get("category_id", 1))  # 1 = balloon

            seg = ann["segmentation"][0]
            poly = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]

            mask_img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask_img).polygon(poly, outline=1, fill=1)
            mask = np.array(mask_img, dtype=np.uint8)
            masks.append(mask)

            areas.append(ann.get("area", float(bw * bh)))
            iscrowd.append(ann.get("iscrowd", 0))

        if not boxes:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.array([], dtype=np.int64)
            masks = np.zeros((0, h, w), dtype=np.uint8)
            areas = np.array([], dtype=np.float32)
            iscrowd = np.array([], dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            masks = np.stack(masks, axis=0)
            areas = np.array(areas, dtype=np.float32)
            iscrowd = np.array(iscrowd, dtype=np.int64)

        img = F.to_tensor(img)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([img_id], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_instance_segmentation_model(num_classes, pretrained=True):
    """
    Shared model builder for training and evaluation.
    Use this in:
      - maskrcnn_train_balloon.py
      - mAp.py / compute_balloon_area.py
    """
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()

        if (i + 1) % 10 == 0:
            print(
                f"[Epoch {epoch}] Step {i+1}/{len(data_loader)} "
                f"Loss: {losses.item():.4f}"
            )

    epoch_loss = running_loss / max(1, len(data_loader))
    print(f"[Epoch {epoch}] Average loss: {epoch_loss:.4f}")
    return epoch_loss


def evaluate(model, data_loader, device):
    model.eval()
    total_images = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            _ = model(images)
            total_images += len(images)
    print(f"Validation: ran on {total_images} images (no metrics computed).")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = BalloonCocoDataset(TRAIN_IMG_DIR, TRAIN_COCO)
    val_dataset   = BalloonCocoDataset(VAL_IMG_DIR, VAL_COCO)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # use shared builder
    model = get_instance_segmentation_model(NUM_CLASSES, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        evaluate(model, val_loader, device)
        print(f"Epoch {epoch} finished in {time.time() - start:.1f}s")

    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print("Saved trained weights to:", OUTPUT_WEIGHTS)


if __name__ == "__main__":
    main()

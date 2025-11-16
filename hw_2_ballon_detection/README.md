# Balloon Detection Benchmark

### YOLO vs Mask R-CNN --- mAP@0.5, Training Summary, and Inference Comparison

This project compares **YOLO** and **Mask R-CNN** on a balloon detection
task using:

-   **Accuracy metric:** mAP@0.5
-   **Models:** **YOLOv8** (Ultralytics) vs. **maskrcnn_resnet50_fpn** (Torchvision)
-   **Aspects compared:** Accuracy, training configuration, inference
    speed on CPU/GPU

## 1. Performance Metric: mAP@0.5

### What is mAP?
mAP (Mean Average Precision) is the standard evaluation metric for object detection. It measures how well the detector identifies and localizes objects. 
mAP is equal to the average of the Average Precision metric across all classes in a model and measured between 0 and 1.

## 2. Model Training Configuration

## 2.1 YOLO (Ultralytics YOLOv8)
| Parameter | Value |
|----------|--------|
| Model | YOLOv8n |
| Epochs | **100** |
| Image size | 640×640 |
| Dataset | YOLO TXT format |
 
## 2.2 Mask R-CNN (Torchvision maskrcnn_resnet50_fpn)

| Parameter  | Value            |
| ---------- | ---------------- |
| Backbone   | ResNet-50 + FPN  |
| Pretrained | COCO             |
| Epochs     | **20**           |
| Batch Size | 2                |
| LR         | 0.005            |
| Dataset    | COCO JSON format |
 
## 3. Why Epoch Counts Differ
I found out that **100 epocsh for YOLOv8 and 20 epochs for Mask R-CNN** would be reasonable values for convergence, so these values were used in this benchmark. 

**YOLOv8 and Mask R-CNN have very different training speeds**
- YOLOv8 is a **single-stage** detector → fast per epoch
- Mask R-CNN is a **two-stage** detector → much slower per epoch

- YOLOv8 typically needs **100–300** epochs and Mask R-CNNconverges within **10–20** epochs


## 4. mAP@0.5 Results

| Model      |mAP@0.5|
|-----------|-------|
|YOLOv8     | 0.3031|
|Mask R-CNN | 0.2075|

## 5. Inference Benchmark
| Model     | GPU (ms/image) | CPU (ms/image)|
|-----------|-----------------|---------------|
|YOLOv8n    | 6.17            |  18.28        |
|Mask R-CNN | 37.28           |  757.02       |


## 6. Which Detector Should You Use?
**Inference results clearrly shows that,**
- YOLOv8 is less computationally intensive and faster than Mask R-CNN but provided too wide bounding boxes for test images.
- Although Mask R-CNN provided more accurate contour for balloons in test images, 
**it has lower mAP@0.5 score**.

### Why Mask R-CNN Looks Better Visually but Achieves Lower mAP@0.5
- According to my research this happens beacuse  mAp@0.5 metric only considers bounding boxes, not masks. Mask R-CNN is primarily a segmentation model, YOLOv8, in contrast, is optimized specifically for bounding-box regression and tends to produce tighter, more consistent boxes, which increases its IoU and therefore its mAP score.**


### Choose Based on Your Needs
**Use YOLOv8 if you need:**
- Real-time speed, lightweight deployment, lower compute usage

**Use Mask R-CNN if you need:**
- Pixel-accurate segmentation, more precise localization, offline, high-quality analysis.

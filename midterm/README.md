# Face & Pose Detection — Direction and Arm-Up Classification

This project uses **MediaPipe Tasks** to perform:
1. **Face direction classification** — Detects faces and determines if each person is looking *left*, *right*, or *straight*.
2. **Pose arm detection** — Detects human poses and determines whether the *left*, *right*, *both*, or *no* arms are raised.

---

## Requirements

```bash
pip install mediapipe opencv-python numpy
or 
conda install -c conda-forge opencv 
```
Ensure you have the following model bundles in your working directory:
- `detector.tflite` – for face detection  
- `pose_landmarker.task` – for pose detection

---
## Pose Arm-Up Detection

### Run
```bash
python pose_arm_detect.py image_file.jpg
```
### Description
- Uses MediaPipe **PoseLandmarker**.  
- Draws full-body landmarks and segmentation mask.  
- For each detected person:
  - Compares wrist y-coordinates with shoulder y-coordinates.
  - Classifies arms as: 
  `both`, `left`, `right`, or `none` is up

### Output
- Prints classification to console.
- Saves annotated image in `./detected_images/` as:
  ```
  pose_<which_arm_up>.png
  ```

### Key Function
```python
output_image, which_arm_up = classify_arm_up(annotated_image, detection_result)
```

## Face Direction Detection

### Run
```bash
python face_direction_detect.py image_file.png
```

### Description
- Uses MediaPipe **FaceDetector**.  
- For each detected face:
  - Draws bounding boxes and facial keypoints.
  - Computes the **nose-tip horizontal offset** relative to the center of the bounding box.  
  - Classifies:
    - `right` → nose offset > +0.12×face width  
    - `left`  → nose offset < −0.12×face width  
    - `straight` otherwise  

### Output
- Prints detected directions to console.
- Saves annotated image in `./detected_images/` as:
  ```
  face_<direction>.png
  ```
### Key Function
```python
annotated_image, directions = classify_look_direction(image, detection_result)
```
- **this function normalizes nose offset by face width to be robust against scale variations.**
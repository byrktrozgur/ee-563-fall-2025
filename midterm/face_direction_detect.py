from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

    # Draw keypoints
    for keypoint in detection.keypoints:
      keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                     width, height)
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    category_name = '' if category_name is None else category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (MARGIN + bbox.origin_x,
                     MARGIN + ROW_SIZE + bbox.origin_y)
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def classify_look_direction(annotated_image, detection_result, thresh_frac=0.12):
    """
    Classify gaze direction for each detected face using FaceDetector keypoints.
    Uses nose-tip horizontal offset inside the face box.
    """
    import cv2
    import numpy as np

    annotated_image = annotated_image.copy()
    H, W = annotated_image.shape[:2]
    directions = []

    for det in getattr(detection_result, "detections", []):
        # Face box in pixels
        bx, by = int(det.bounding_box.origin_x), int(det.bounding_box.origin_y)
        bw, bh = int(det.bounding_box.width),  int(det.bounding_box.height)
        cx = bx + bw * 0.5 # center x

        # Default if keypoints missing
        direction = "straight"

        # FaceDetector keypoints: [right_eye, left_eye, nose_tip, mouth, right_ear, left_ear]
        if det.keypoints and len(det.keypoints) >= 3:
            nose = det.keypoints[2]  # nose tip
            nose_x_px = nose.x * W  # nose.x --> [0,1]

            # Normalize by face width for a scale-invariant offset
            offset = (nose_x_px - cx) / max(bw, 1) # prevent div by zero
            # offset is around [-0.5, 0.5] so 0.10 ~ 15 can be a reasonable threshold
            if offset > thresh_frac:
                direction = "right" # 
            elif offset < -thresh_frac:
                direction = "left" 
            else:
                direction = "straight"

        directions.append(direction)

        # Draw the direction text near the face box
        cv2.putText(annotated_image, direction, (bx, max(0, by - 8)),
                    cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 0, 255), 2)
        
        return annotated_image, directions

import cv2
import sys

if len(sys.argv) < 2:
  print('usage = python face_direction_detect.py image_file.png')
  sys.exit(1)

IMAGE_PATH = './test_images/'
user_input = sys.argv[1]
IMAGE_FILE = IMAGE_PATH + user_input
image = cv2.imread(IMAGE_FILE)
if image is None:
  print(f"Could not read image file: {IMAGE_FILE}")
  sys.exit(1)

# STEP 1: Import the necessary modules.
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
mp_image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect faces in the input image.
detection_result = detector.detect(mp_image)
# print(detection_result)

# STEP 5: Classify looking  direction for each detected face.
image_copy = np.copy(mp_image.numpy_view())
annotated_image = visualize(image_copy, detection_result)

output_image, dirs = classify_look_direction(annotated_image, detection_result, thresh_frac=0.12)

direction = dirs[0]
print(dirs)
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(f"./detected_images/face_{direction}.png", output_image_bgr)




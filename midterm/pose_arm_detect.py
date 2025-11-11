from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

#wget -q -O image.jpg https://cdn.pixabay.com/photo/2019/03/12/20/39/girl-4051811_960_720.jpg
def classify_arm_up(annotated_image, detection_result, margin_px: int = 10) -> str:
    """
    Classify which arm is up using MediaPipe Tasks PoseLandmarker output.
    Returns:
      One of {"left","right","both","None"} for the first detected person.
    """
    h = annotated_image.shape[0]
    poses = getattr(detection_result, "pose_landmarks", [])
    if not poses:
        return "None"

    which_arm_up = "None"
    # MediaPipe Pose indices
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_WRIST,   RIGHT_WRIST   = 15, 16

    lm = poses[0]  # use first person
    y = lambda i: int(lm[i].y * h)  # normalized -> pixels

    left_up  = y(LEFT_WRIST)  < y(LEFT_SHOULDER) - margin_px
    right_up = y(RIGHT_WRIST) < y(RIGHT_SHOULDER) - margin_px
    
    if left_up and right_up:
      which_arm_up = "both"
    elif left_up:
      which_arm_up = "left"
    elif right_up:
      which_arm_up = "right"

    cv2.putText(annotated_image, f"Arm up: {which_arm_up}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  
    return annotated_image, which_arm_up


IMAGE_FILE = './test_images/pose-3.jpg'
import cv2
import sys

if len(sys.argv) < 2:
  print('usage = python pose_arm_detect.py image_file.jpg')
  sys.exit(1)

IMAGE_PATH = './test_images/'
user_input = sys.argv[1]
IMAGE_FILE = IMAGE_PATH + user_input
image = cv2.imread(IMAGE_FILE)
if image is None:
  print(f"Could not read image file: {IMAGE_FILE}")
  sys.exit(1)
  
# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
mp_image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(mp_image)

# STEP 5: Process the detection result.
annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
""" # save annotated image to a file
OUTPUT_FILE = 'pose_landmarker_image_output.png'
cv2.imwrite(OUTPUT_FILE, rgb_annotated_image)
print(f"Annotated image saved to {OUTPUT_FILE}") """

segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
""" # save segmentation mask to a file
MASK_OUTPUT_FILE = 'pose_landmarker_segmentation_mask.png'
cv2.imwrite(MASK_OUTPUT_FILE, visualized_mask)
print(f"Segmentation mask saved to {MASK_OUTPUT_FILE}") """

output_image,which_arm_up = classify_arm_up(annotated_image, detection_result, margin_px=10)

if which_arm_up == "both":
    print("Both arms are up.")
elif which_arm_up == "None":
    print("No arm is up.")
else:
  print(f"{which_arm_up} arm is up.")

# save the annotated image with arm info
output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
OUTPUT_FILE = f"./detected_images/{which_arm_up}_arm_up.png"
cv2.imwrite(OUTPUT_FILE, output_image_bgr)
print(f"Annotated image with arm info saved to {OUTPUT_FILE}")

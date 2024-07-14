import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

import os
import supervision as sv
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
finger_coords = {}

# Base option setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Hand Landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# Pose Landmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
body_task_file = os.path.dirname(os.path.realpath(__file__))  + '/../../../models/pose_landmarker.task'



def handle_result_pose(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
        Handles the result of model by saving it into the global `current_result`
        :param result: the result obtained

        :param timestamp_ms: IGNORED (needed for calling convention)
        :param output_image: IGNORED (needed for calling convention)
        """
    current_pose_result.clear()
    if len(result.pose_landmarks) > 0:
        current_pose_result.append(result)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=body_task_file),
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result_pose
)


current_pose_result: list[PoseLandmarkerResult] = []

def handle_result_pose(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
        Handles the result of model by saving it into the global `current_result`
        :param result: the result obtained

        :param timestamp_ms: IGNORED (needed for calling convention)
        :param output_image: IGNORED (needed for calling convention)
        """
    current_pose_result.clear()
    if len(result.pose_landmarks) > 0:
        current_pose_result.append(result)

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
def store_finger_node_coords(id: int, cx: float, cy: float):
  ''' Function takes in a node id, the x and y position of the node.
  
    Stores the position in a list of positions with each index representing a frame
    from the video. The list for each node is acquired by using a dictionary with the id
    as a key.'''
  if id not in finger_coords:
    finger_coords[id] = []
  finger_coords[id].append((cx, cy))

def get_position(lm: NormalizedLandmark, width: int, height: int) -> tuple[int, int]:
    """
    Maps a landmark to a pixel
    :param lm the landmark to use (with x,y in [0,1])
    :param width: the image width
    :param height: the image height
    :return: the pixel position of the landmark
    """
    x = int(width * lm.x)
    y = int(height * lm.y)
    return x, y

model = YOLO('/Users/felixlu/Desktop/Evaluator/Evaluator-code/src/computer_vision/hand_pose_detection/best-2 1.pt')  # Path to your model file
# For webcam input:
# model.overlap = 80
video_file_path = '/Users/felixlu/Desktop/Evaluator/Evaluator-code/src/computer_vision/hand_pose_detection/Too much pronation (1).mp4'
cap = cv2.VideoCapture(video_file_path)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, PoseLandmarker.create_from_options(pose_options) as pose:
  
  frame_count = 0
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
      
    frame_count += 1
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    pose.detect_async(mp_image, frame_count)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    # pose_results = pose.process(image)
    hand_node_positions = []
    # yolov8 prediction
    YOLOresults = model(image)
    detections = sv.Detections.from_ultralytics(YOLOresults[0])
    # Draw the hand + pose annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape
    height = len(image)
    width = len(image[0])
    if results.multi_hand_landmarks:

      if len(current_pose_result) != 0:
        r = current_pose_result[0]
        for cur in r.pose_landmarks:
          landmarks = [get_position(x, width, height) for x in cur]
          for lm in landmarks[11:15] + landmarks[23:29]: # excludes the face and hand pose landmarks
              image = cv2.circle(image, lm, radius=10, color=(255, 0, 0), thickness=-1)


      for hand_landmarks in results.multi_hand_landmarks:
        # node_positions = []
        for ids, landmrk in enumerate(hand_landmarks.landmark):
          # print(ids, landmrk)
          cx, cy = landmrk.x * image_width, landmrk.y*image_height

          # calls function to store position of nodes
          store_finger_node_coords(id, cx, cy)
          # print("id:", ids, " x:", cx, " y:", cy)
          # print("id type: ", type(ids), " x type: ", type(cx), " y type: ", type(cy))
          print (ids, cx, cy)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        #mp_drawing.draw_landmarks(
            #image,
            #pose_results.pose_landmarks,
            #None,
            #mp_drawing.DrawingSpec(color=(255,0,0), thickness=4, circle_radius=8))

    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(
        scene=image,
        detections=detections
    )

    # print("Annotated Frame: ", annotated_frame)
    # Flip the image horizontally for a selfie-view display.
    # cv2.namedWindow('Mediapipe Hands', cv2.WINDOW_NORMAL)

    image = ResizeWithAspectRatio(image, height=800)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
# print("Hand node Values: ", hand_node_positions)
# print(finger_coords)

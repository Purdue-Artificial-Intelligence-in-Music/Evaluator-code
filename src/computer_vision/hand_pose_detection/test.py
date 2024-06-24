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
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
finger_coords = {}


def store_finger_node_coords(id: int, cx: float, cy: float):
  ''' Function takes in a node id, the x and y position of the node.
  
    Stores the position in a list of positions with each index representing a frame
    from the video. The list for each node is acquired by using a dictionary with the id
    as a key.'''
  if id not in finger_coords:
    finger_coords[id] = []
  finger_coords[id].append((cx, cy))

model = YOLO('best.pt')  # Path to your model file
# For webcam input:
video_file_path = 'Too much pronation (1).mp4'
cap = cv2.VideoCapture(video_file_path)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    hand_node_positions = []
    # yolov8 prediction
    YOLOresults = model(image)
    detections = sv.Detections.from_ultralytics(YOLOresults[0])
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        node_positions = []
        for ids, landmrk in enumerate(hand_landmarks.landmark):
          # print(ids, landmrk)
          cx, cy = landmrk.x * image_width, landmrk.y*image_height

          # calls function to store position of nodes
          store_finger_node_coords(id, cx, cy)
          # print("id:", ids, " x:", cx, " y:", cy)
          # print("id type: ", type(ids), " x type: ", type(cx), " y type: ", type(cy))
          # print (ids, cx, cy)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    
    # add bounding boxes
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(
        scene=image,
        detections=detections
    )
    # print("Annotated Frame: ", annotated_frame)
    # Flip the image horizontally for a selfie-view display.
    # cv2.namedWindow('Mediapipe Hands', cv2.WINDOW_NORMAL)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
# print("Hand node Values: ", hand_node_positions)
print(finger_coords)
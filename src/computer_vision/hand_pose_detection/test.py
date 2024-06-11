import cv2
import mediapipe as mp
import numpy as np
import os
import supervision as sv
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = YOLO('/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/best.pt')  # Path to your model file
# For webcam input:
video_file_path = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Wrong bow hold.mp4'
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

    # yolov8 prediction
    YOLOresults = model(image)
    detections = sv.Detections.from_ultralytics(YOLOresults[0])

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
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

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
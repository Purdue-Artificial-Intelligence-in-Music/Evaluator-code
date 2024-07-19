import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import os
import supervision as sv
import ultralytics
from ultralytics import YOLO
import torch

# Gesture model for hands

# option setup for gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# gesture model path (set path to gesture_recognizer_custom.task)
gesture_model = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/gesture_recognizer_custom.task'

# A class that stores methods/data for 2d points on the screen

class Point2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def distance_to(self, other):
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def as_tuple(self):
        return (self.x, self.y)
    
    def find_point_p1(A, B, ratio=0.7):
        """
        Finds the coordinates of point P1 that is `ratio` distance from A to B.
        
        Parameters:
        A (Point2D): Point A
        B (Point2D): Point B
        ratio (float): Ratio of the distance from A to B where P1 should be (default is 0.7)
        
        Returns:
        Point2D: Coordinates of point P1
        """
        Px = A.x + ratio * (B.x - A.x)
        Py = A.y + ratio * (B.y - A.y)
        return Point2D(Px, Py)

# Function to resize image with aspect ratio
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

# Function to store finger node coordinates
def store_finger_node_coords(id: int, cx: float, cy: float, finger_coords: dict):
    if id not in finger_coords:
        finger_coords[id] = []
    finger_coords[id].append((cx, cy))


def main():
  # YOLOv8 model trained from Roboflow dataset
  # Used for bow and target area oriented bounding boxes
  model = YOLO('/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/best-2 1.pt')  # Path to your model file
  
  # For webcam input:
  # model.overlap = 80

  #input video file
  video_file_path = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Vertigo for Solo Cello - Cicely Parnas.mp4'
  cap = cv2.VideoCapture(video_file_path) # change argument to 0 for demo/camera input

  frame_count = 0
  output_frame_length = 960
  output_frame_width = 720

  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  finger_coords = {}

  # Initialize video writer
  output_file = 'output.mp4' 
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  # saves output video to output_file
  writer = cv2.VideoWriter(output_file, fourcc, 12.5, (output_frame_length, output_frame_width))

  #setup gesture options
  num_hands = 2
  gesture_options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=gesture_model),
    running_mode=VisionRunningMode.VIDEO,
    num_hands = num_hands)

  with mp_hands.Hands(
      model_complexity=0,
      max_num_hands=num_hands,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands, mp_pose.Pose(
      model_complexity=0,
      min_detection_confidence=0.4,
      min_tracking_confidence=0.6) as pose, GestureRecognizer.create_from_options(gesture_options) as recognizer:


    writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(output_frame_length,output_frame_width)) # algo makes a frame every ~80ms = 12.5 fps
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        break
  
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)
      pose_results = pose.process(image)
      hand_node_positions = []

      # gesture classification data arrays
      current_gestures = []
      current_handedness = []
      current_score = []

      # recognize gestures
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
      gesture_recognition_result = recognizer.recognize_for_video(mp_image, frame_count)
      frame_count += 1

      # obtain neccesary data into array for display (using array because there are two hands)
      if gesture_recognition_result is not None and any(gesture_recognition_result.gestures):
        print("Recognized gestures:")
        for single_hand_gesture_data in gesture_recognition_result.gestures:
          gesture_name = single_hand_gesture_data[0].category_name
          current_gestures.append(gesture_name)

        for single_hand_handedness_data in gesture_recognition_result.handedness:
          hand_name = single_hand_handedness_data[0].category_name
          current_handedness.append(hand_name)

        for single_hand_score_data in gesture_recognition_result.gestures:
          score = single_hand_score_data[0].score
          current_score.append(round(score, 2))

      # display classified gesture data on frames
      y_pos = image.shape[0] - 70
      for x in range(len(current_gestures)):
        txt = current_handedness[x] + ": " + current_gestures[x] + " " + str(current_score[x])
        cv2.putText(image, txt, (image.shape[1] - 400, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (173,216,230), 2, cv2.LINE_AA)
        print(txt)
        y_pos += 50

      YOLOresults = model(image)
      for result in YOLOresults:
        if len(result.obb.xyxyxyxy) > 0:
          coord_box_one = result.obb.xyxyxyxy[0]
          round_coord_box_one = torch.round(coord_box_one)

          box_one_coordinate_1 = round_coord_box_one[0]  # First coordinate (x1, y1)
          box_two_coordinate_2 = round_coord_box_one[1]   # Second coordinate (x2, y2)
          box_three_coordinate_3 = round_coord_box_one[2]   # Third coordinate (x3, y3)
          box_four_coordinate_4 = round_coord_box_one[3]   # Fourth coordinate (x4, y4)

          x1, y1 = box_one_coordinate_1[0].item(), box_one_coordinate_1[1].item()
          x2, y2 = box_two_coordinate_2[0].item(), box_two_coordinate_2[1].item()
          x3, y3 = box_three_coordinate_3[0].item(), box_three_coordinate_3[1].item()
          x4, y4 = box_four_coordinate_4[0].item(), box_four_coordinate_4[1].item()

          # Print rounded coordinates
          print("coord box one", round_coord_box_one)
          print("first coord = ", coord_box_one)

          # Prepare text
          text_one = "Bow OBB Coords:"
          text_coord1 = f"Coord 1: ({x1}, {y1})"
          text_coord2 = f"Coord 2: ({x2}, {y2})"
          text_coord3 = f"Coord 3: ({x3}, {y3})"
          text_coord4 = f"Coord 4: ({x4}, {y4})"

          # Define bottom left corners for each text line
          bottom_left_corner_text_one = (20, image.shape[0] - 140)  # Adjusted to move higher
          bottom_left_corner_coord1 = (20, image.shape[0] - 110)   # Adjusted to move higher
          bottom_left_corner_coord2 = (20, image.shape[0] - 80)    # Adjusted to move higher
          bottom_left_corner_coord3 = (20, image.shape[0] - 50)    # Adjusted to move higher
          bottom_left_corner_coord4 = (20, image.shape[0] - 20)    # Adjusted to move higher

          # Put text on image for box one
          cv2.putText(image, text_one, bottom_left_corner_text_one, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (48, 198, 26), 4)
          cv2.putText(image, text_coord1, bottom_left_corner_coord1, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (48, 198, 26), 4)
          cv2.putText(image, text_coord2, bottom_left_corner_coord2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (48, 198, 26), 4)
          cv2.putText(image, text_coord3, bottom_left_corner_coord3, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (48, 198, 26), 4)
          cv2.putText(image, text_coord4, bottom_left_corner_coord4, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (48, 198, 26), 4)

        if len(result.obb.xyxyxyxy) >= 2:
          coord_box_two = result.obb.xyxyxyxy[1]
          round_coord_box_two = torch.round(coord_box_two)

          box_two_coordinate_1 = round_coord_box_two[0]  # First coordinate (x1, y1)
          box_two_coordinate_2 = round_coord_box_two[1]  # Second coordinate (x2, y2)
          box_two_coordinate_3 = round_coord_box_two[2]  # Third coordinate (x3, y3)
          box_two_coordinate_4 = round_coord_box_two[3]  # Fourth coordinate (x4, y4)

          # Extract individual points for box two
          x1_2, y1_2 = box_two_coordinate_1[0].item(), box_two_coordinate_1[1].item()
          x2_2, y2_2 = box_two_coordinate_2[0].item(), box_two_coordinate_2[1].item()
          x3_2, y3_2 = box_two_coordinate_3[0].item(), box_two_coordinate_3[1].item()
          x4_2, y4_2 = box_two_coordinate_4[0].item(), box_two_coordinate_4[1].item()

          # Prepare text for box one
          text_two = "TA OBB Coords:"
          text_coord1_2 = f"Coord 1: ({x1_2}, {y1_2})"
          text_coord2_2 = f"Coord 2: ({x2_2}, {y2_2})"
          text_coord3_2 = f"Coord 3: ({x3_2}, {y3_2})"
          text_coord4_2 = f"Coord 4: ({x4_2}, {y4_2})"
          text_offset = 30  # spacing between lines
          top_right_corner_text_two = (image.shape[1] - 550, text_offset + 20) # Adjusted to move down and left
          top_right_corner_coord1_2 = (image.shape[1] - 550, text_offset * 2 + 20) # Adjusted to move down and left
          top_right_corner_coord2_2 = (image.shape[1] - 550, text_offset * 3 + 20) # Adjusted to move down and left
          top_right_corner_coord3_2 = (image.shape[1] - 550, text_offset * 4 + 20) # Adjusted to move down and left
          top_right_corner_coord4_2 = (image.shape[1] - 550, text_offset * 5 + 20) # Adjusted to move down and left

          # Put text on image for box two
          cv2.putText(image, text_two, top_right_corner_text_two, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)
          cv2.putText(image, text_coord1_2, top_right_corner_coord1_2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)
          cv2.putText(image, text_coord2_2, top_right_corner_coord2_2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)
          cv2.putText(image, text_coord3_2, top_right_corner_coord3_2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)
          cv2.putText(image, text_coord4_2, top_right_corner_coord4_2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)

      detections = sv.Detections.from_ultralytics(YOLOresults[0])

      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image_height, image_width, _ = image.shape

      if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
              for ids, landmrk in enumerate(hand_landmarks.landmark):
                  cx, cy = landmrk.x * image_width, landmrk.y * image_height
                  store_finger_node_coords(ids, cx, cy, finger_coords)
              mp_drawing.draw_landmarks(
                  image,
                  hand_landmarks,
                  mp_hands.HAND_CONNECTIONS,
                  mp_drawing_styles.get_default_hand_landmarks_style(),
                  mp_drawing_styles.get_default_hand_connections_style())

              landmark_subset = landmark_pb2.NormalizedLandmarkList(
                  landmark=pose_results.pose_landmarks.landmark[11:15]
              )
              mp_drawing.draw_landmarks(
                  image,
                  landmark_subset,
                  None,
                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=10, circle_radius=6))

      oriented_box_annotator = sv.OrientedBoxAnnotator()
      annotated_frame = oriented_box_annotator.annotate(
          scene=image,
          detections=detections
      )

      image = ResizeWithAspectRatio(image, height=800)
      image = cv2.putText(
          image,
          "Frame {}".format(frame_count),
          (10, 50),
          cv2.QT_FONT_NORMAL,
          1,
          (0, 0, 255),
          1,
          cv2.LINE_AA
      )
      
      # Resize to specified output dimensions before writing
      resized_frame = cv2.resize(image, (output_frame_length, output_frame_width))

      writer.write(resized_frame)
      cv2.imshow('MediaPipe Hands', image)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

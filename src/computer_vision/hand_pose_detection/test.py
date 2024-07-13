import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import os
import supervision as sv
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

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

def wrap_text(text, max_width):
  import textwrap
  wrapper = textwrap.TextWrapper(width=max_width)
  wrapped_text = wrapper.fill(text)
  return wrapped_text

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
  finger_coords = {}


  ''' Function takes in a node id, the x and y position of the node.
  
    Stores the position in a list of positions with each index representing a frame
    from the video. The list for each node is acquired by using a dictionary with the id
    as a key.'''
  if id not in finger_coords:
    finger_coords[id] = []
  finger_coords[id].append((cx, cy))

def main():
  model = YOLO('/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/best-2 1.pt')  # Path to your model file
  # For webcam input:
  # model.overlap = 80

  #input file
  video_file_path = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Too much pronation (1).mp4'
  cap = cv2.VideoCapture(video_file_path)

  frame_count = 0

  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands

  with mp_hands.Hands(
      model_complexity=0,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as hands, mp_pose.Pose(
      model_complexity=0,
      min_detection_confidence=0.4,
      min_tracking_confidence=0.6) as pose:


    writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(640,480)) # algo makes a frame every ~80ms = 12.5 fps
    while cap.isOpened():
      success, image = cap.read()
      frame_count += 1
      #if not success:
        #print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        #break
  
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = hands.process(image)
      pose_results = pose.process(image)
      hand_node_positions = []
      # yolov8 prediction
      YOLOresults = model(image)
      for result in YOLOresults:
          obb = result.obb
          # if obb is not None:
          #   coords = obb.xyxyxyxy
          #   print("Box 1:", obb.xyxyxyxy)
          #   print("Box 2:", (obb.xyxyxyxy)[0])
          text = ""
          # for i in range(len(result.obb.xyxyxyxy)):
          #   coords = result.obb.xyxyxyxy[i]  # Accessing xyxyxyxy property to get the coordinates
          #   # text = text + "\n" + f"Box {i+1}: ({coords})"
          #   coords_str = ", ".join(map(str, coords))
          #   text += f"Box {i + 1}: ({coords_str})\n"
          #   print(f"Box {i + 1} coordinates:")
          #   print(coords)
          # max_text_width = 100  # Adjust based on your image and font size
          # wrapped_text = wrap_text(text, max_text_width)
          
          if (len(result.obb.xyxyxyxy) > 0):
            coord_box_one = result.obb.xyxyxyxy[0]
            text_one = f"Bow OBB coords: ({coord_box_one})"
            print("Bow OBB coords:", coord_box_one)
            bottom_left_corner = (20, image.shape[0] - 30)
            cv2.putText(image, text_one,  bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (98, 0, 158), 4)
          if (len(result.obb.xyxyxyxy) >= 2):
            coord_box_two = result.obb.xyxyxyxy[1]
            # TA (Target Area)
            text_two = f"TA OBB coords: ({coord_box_two})"
            print("TA OBB coords", coord_box_two)
            top_right_corner = (20, image.shape[1] + 100)
            cv2.putText(image, text_two, top_right_corner, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 4)


      # for result in YOLOresults:
      #   if result.obb:
      #     obb = result.obb
      #     for i in range(len(obb.xyxyxyxy)):
      #       coords = obb.xyxyxyxy[i]
      #       # Format coordinates text
      #       text = f"Box {i + 1}: ({coords[0]}, {coords[1]}), ({coords[2]}, {coords[3]}), ({coords[4]}, {coords[5]}), ({coords[6]}, {coords[7]})"
      #       # Draw text on the image
      #     cv2.putText(image, text, (20, image.shape[0] - 30 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
      #     #  print("Box 2:", obb.xyxyxyxy[1])
      #     #  print(obb.id)
      detections = sv.Detections.from_ultralytics(YOLOresults[0])
      # print("Detections:", detections)

  # Iterate over each detection
      # count = 0
      # for detection in detections:
      #     count += 1
      #     # print(f"Detection {count}: {detection}")
      #     # print("Detection structure:", type(detection), detection)

      #     # Ensure detection is iterable and extract coordinates, confidence, and class
      #     try:
      #         xyxy = detection[0]  # Bounding box coordinates
      #         conf = detection[2]  # Confidence score
      #         cls = detection[3]  # Class ID
              
      #         # Ensure xyxy is a list or array before converting to int
      #         xmin, ymin, xmax, ymax = map(int, xyxy)
      #         print(f"Bounding Box {count}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
      #         text = f'({xmin},{ymin}),({xmax},{ymax})'
      #         cv2.putText(image, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      #     except Exception as e:
      #         print(f"Error processing detection {count}: {e}")
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

          landmark_subset = landmark_pb2.NormalizedLandmarkList(
            landmark = pose_results.pose_landmarks.landmark[11:15]
          )
          mp_drawing.draw_landmarks(
              image,
              landmark_subset,
              None,
              mp_drawing.DrawingSpec(color=(255,0,0), thickness=10, circle_radius=6))

      oriented_box_annotator = sv.OrientedBoxAnnotator()
      annotated_frame = oriented_box_annotator.annotate(
          scene=image,
          detections=detections
      )
  
      #print("Annotated Frame: ", annotated_frame)
      # Flip the image horizontally for a selfie-view display.
      # cv2.namedWindow('Mediapipe Hands', cv2.WINDOW_NORMAL)
  
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
      writer.write(image)
      cv2.imshow('MediaPipe Hands', image)
      #if cv2.waitKey(5) & 0xFF == 27:
        #break
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  cap.release()
  cv2.destroyAllWindows()
  # print("Hand node Values: ", hand_node_positions)
  # print(finger_coords)

if __name__ == "__main__":
    main()
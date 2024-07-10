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
  ''' Function takes in a node id, the x and y position of the node.
  
    Stores the position in a list of positions with each index representing a frame
    from the video. The list for each node is acquired by using a dictionary with the id
    as a key.'''
  if id not in finger_coords:
    finger_coords[id] = []
  finger_coords[id].append((cx, cy))
model = YOLO('best-2 1.pt')  # Path to your model file
# For webcam input:
# model.overlap = 80
video_file_path = 'Too much pronation (1).mp4'
cap = cv2.VideoCapture(video_file_path)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands, mp_pose.Pose(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
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
        for i in range(len(result.obb.xyxyxyxy)):
          coords = result.obb.xyxyxyxy[i]  # Accessing xyxyxyxy property to get the coordinates
          # text = text + "\n" + f"Box {i+1}: ({coords})"
          coords_str = ", ".join(map(str, coords))
          text += f"Box {i + 1}: ({coords_str})\n"
          print(f"Box {i + 1} coordinates:")
          print(coords)
        max_text_width = 100  # Adjust based on your image and font size
        wrapped_text = wrap_text(text, max_text_width)
        bottom_left_corner = (20, image.shape[0] - 30)
        cv2.putText(image, text,  bottom_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


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
        
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            None,
            mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4))
 
    oriented_box_annotator = sv.OrientedBoxAnnotator()
    annotated_frame = oriented_box_annotator.annotate(
        scene=image,
        detections=detections
    )
 
    #print("Annotated Frame: ", annotated_frame)
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
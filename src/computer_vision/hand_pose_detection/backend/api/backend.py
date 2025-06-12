import base64
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import os
from pathlib import Path
import supervision as sv
import ultralytics
from ultralytics import YOLO
import torch
from PIL import Image

from datetime import datetime

# Gesture model for hands

# option setup for gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# gesture model path (set path to gesture_recognizer_custom.task)
hand_pose_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
gesture_model = os.path.join(hand_pose_dir, "gesture_recognizer_custom.task")

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
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y}
    
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

    def find_intersection(p1, p2, p3, p4):
        # Line 1: passing through p1 and p2
        A1 = p2.y - p1.y  # y2 - y1
        B1 = p1.x - p2.x  # x1 - x2
        C1 = A1 * p1.x + B1 * p1.y
 
        # Line 2: passing through p3 and p4
        A2 = p4.y - p3.y  # y4 - y3
        B2 = p3.x - p4.x  # x3 - x4
        C2 = A2 * p3.x + B2 * p3.y
 
        # Determinant of the system
        det = A1 * B2 - A2 * B1
 
        if det == 0:
            # Lines are parallel (no intersection)
            return None
        else:
            # Lines intersect, solving for x and y
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
        return Point2D(x, y)
    
    def is_above_or_below(self, A, B):
        """
        Determines if the current point (self) is above or below the line segment defined by points A and B.
        Parameters:
        A (Point2D): First endpoint of the line segment.
        B (Point2D): Second endpoint of the line segment.
        Returns:
        bool: True if the current point (self) is above the line, False if it is below or on the line.
        """
        # Calculate the cross product of vectors AB and AC (where C is self)
        cross_product = (B.x - A.x) * (self.y - A.y) - (B.y - A.y) * (self.x - A.x)
        if cross_product > 0:
            return True  # Current point (self) is above the line
        else:
            return False 
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


def videoFeed(video_path_arg, output_path):
    # YOLOv8 model trained from Roboflow dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))  # /backend/api
    parent_dir = os.path.dirname(os.path.dirname(current_dir))
    # load YOLO model
    model_dir = os.path.join(parent_dir, "bow_target.pt")
    print(f"Looking for model at: {model_dir}")
    model = YOLO(model_dir)
    
    # video capture setup
    cap = cv2.VideoCapture(video_path_arg)
    if not cap.isOpened():
        raise Exception("Failed to open video file")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    output_frame_length = 960
    output_frame_width = 720

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_frame_length, output_frame_width))
    if not out.isOpened():
        raise Exception("Failed to create output video file")
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    finger_coords = {}


    #setup gesture options
    num_hands = 2
    gesture_options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_buffer=open(gesture_model, "rb").read()),
        running_mode=VisionRunningMode.VIDEO,
        num_hands = num_hands)
  
    num_none = 0
    num_supination = 0
    num_correct = 0
    display_gesture = "none"

    desired_fps = 30 
    frame_delay = int(1000 / desired_fps)

    #set up hands and body
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6) as pose, GestureRecognizer.create_from_options(gesture_options) as recognizer:

    
        # writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(output_frame_length,output_frame_width)) # algo makes a frame every ~80ms = 12.5 fps

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

            # update gesture classifcation every 15 frames
            if (frame_count % 30 == 0):
                if max(num_correct, num_none, num_supination) == num_supination:
                    display_gesture = "supination"
                elif max(num_correct, num_none, num_supination) == num_correct:
                    display_gesture = "correct"
                else:
                    display_gesture = "none"
                num_none = 0
                num_supination = 0
                num_correct = 0
            
            # obtain neccesary data into array for display (using array because there are two hands)
            if gesture_recognition_result is not None and any(gesture_recognition_result.gestures):
                for single_hand_gesture_data in gesture_recognition_result.gestures:
                    gesture_name = single_hand_gesture_data[0].category_name
                    current_gestures.append(gesture_name)

                for single_hand_handedness_data in gesture_recognition_result.handedness:
                    hand_name = single_hand_handedness_data[0].category_name
                    current_handedness.append(hand_name)

                for single_hand_score_data in gesture_recognition_result.gestures:
                    score = single_hand_score_data[0].score
                    current_score.append(round(score, 2))

            y_pos = image.shape[0] - 70
            for x in range(len(current_gestures)):
                if current_handedness[x] != "Left":
                    # increment number of none/supination for past 10 frames
                    if current_gestures[x] == "supination":
                        num_supination += 1
                    elif current_gestures[x] == "correct":
                        num_correct += 1
                    else:
                        num_none += 1
            
                    # display classified gesture data on frames
                    txt = current_handedness[x] + ": " + display_gesture + " " + str(current_score[x])
                    if (display_gesture == "supination"):
                        cv2.putText(image, txt, (image.shape[1] - 600, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (218,10,3), 4, cv2.LINE_AA)
                    else:
                        cv2.putText(image, txt, (image.shape[1] - 650, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (37,245,252), 4, cv2.LINE_AA)

            bow_coord_list = []
            string_coord_list =[]
            YOLOresults = model(image)
            for result in YOLOresults:
                if len(result.obb.xyxyxyxy) > 0:
                    coord_box_one = result.obb.xyxyxyxy[0]
                    round_coord_box_one = torch.round(coord_box_one)

                    box_str_coordinate_1 = round_coord_box_one[0]  # First coordinate (x1, y1)
                    box_str_coordinate_2 = round_coord_box_one[1]   # Second coordinate (x2, y2)
                    box_str_coordinate_3 = round_coord_box_one[2]   # Third coordinate (x3, y3)
                    box_str_coordinate_4 = round_coord_box_one[3]   # Fourth coordinate (x4, y4)

                    #create Point2D objects for each box coordinate
                    box_str_point_one = Point2D(box_str_coordinate_1[0].item(), box_str_coordinate_1[1].item())
                    box_str_point_two = Point2D(box_str_coordinate_2[0].item(), box_str_coordinate_2[1].item())
                    box_str_point_three = Point2D(box_str_coordinate_3[0].item(), box_str_coordinate_3[1].item())
                    box_str_point_four = Point2D(box_str_coordinate_4[0].item(), box_str_coordinate_4[1].item())

                    # Prepare text
                    text_one = "String OBB Coords:"
                    text_coord1 = f"Coord 1: ({box_str_point_one.x}, {box_str_point_one.y})"
                    text_coord2 = f"Coord 2: ({box_str_point_two.x}, {box_str_point_two.y})"
                    text_coord3 = f"Coord 3: ({box_str_point_three.x}, {box_str_point_three.y})"
                    text_coord4 = f"Coord 4: ({box_str_point_four.x}, {box_str_point_four.y})"


                    # Define the color and size of the dot
                    radius = 5           # Radius of the dot
                    thickness = -1       # Thickness -1 fills the circle, creating a dot

                    #SHOWING DOTS
                    cv2.circle(image, (int(box_str_point_one.x), int(box_str_point_one.y)), radius, (255, 0, 0), thickness) # bottom left
                    cv2.circle(image, (int(box_str_point_two.x), int(box_str_point_two.y)), radius, (0, 0, 0), thickness) # bottom right
                    cv2.circle(image, (int(box_str_point_three.x), int(box_str_point_three.y)), radius, (0, 255, 0), thickness) # top right
                    cv2.circle(image, (int(box_str_point_four.x), int(box_str_point_four.y)), radius, (0, 0, 255), thickness) # top left
                    string_coord_list.append(box_str_point_one)
                    string_coord_list.append(box_str_point_two)
                    string_coord_list.append(box_str_point_three)
                    string_coord_list.append(box_str_point_four)
                    # Define bottom left corners for each text line
                    bottom_left_corner_text_one = (image.shape[1] - 370, 35 * 6 + 20)  # Adjusted to move higher
                    bottom_left_corner_coord1 = (image.shape[1] - 370, 35 * 7 + 15)   # Adjusted to move higher
                    bottom_left_corner_coord2 = (image.shape[1] - 370, 35 * 8 + 10)    # Adjusted to move higher
                    bottom_left_corner_coord3 = (image.shape[1] - 370, 35 * 9 + 5)    # Adjusted to move higher
                    bottom_left_corner_coord4 = (image.shape[1] - 370, 35 * 10 + 0)    # Adjusted to move higher
                    # Put text on image for box one
                    cv2.putText(image, text_one, bottom_left_corner_text_one, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(image, text_coord1, bottom_left_corner_coord1, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(image, text_coord2, bottom_left_corner_coord2, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(image, text_coord3, bottom_left_corner_coord3, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(image, text_coord4, bottom_left_corner_coord4, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
            
                    # CALCULATING P1
                    #pointOne = Point2D.find_point_p1(leftCoordOne, rightCoordTwo, ratio=0.7)


                if len(result.obb.xyxyxyxy) >= 2:
                    coord_box_two = result.obb.xyxyxyxy[1]
                    round_coord_box_two = torch.round(coord_box_two)

                    box_bow_coordinate_1 = round_coord_box_two[0]  # First coordinate (x1, y1)
                    box_bow_coordinate_2 = round_coord_box_two[1]  # Second coordinate (x2, y2)
                    box_bow_coordinate_3 = round_coord_box_two[2]  # Third coordinate (x3, y3)
                    box_bow_coordinate_4 = round_coord_box_two[3]  # Fourth coordinate (x4, y4)

                    # Define the color and size of the dot
                    radius = 5           # Radius of the dot
                    thickness = -1       # Thickness -1 fills the circle, creating a dot
                    # Add the dot to the image at the specified coordinates
                    box_bow_coord_one = Point2D(box_bow_coordinate_1[0].item(), box_bow_coordinate_1[1].item())
                    box_bow_coord_two = Point2D(box_bow_coordinate_2[0].item(), box_bow_coordinate_2[1].item())
                    box_bow_coord_three = Point2D(box_bow_coordinate_3[0].item(), box_bow_coordinate_3[1].item())
                    box_bow_coord_four = Point2D(box_bow_coordinate_4[0].item(), box_bow_coordinate_4[1].item())
                    # SHOWING DOTS
                    cv2.circle(image, (int(box_bow_coord_one.x), int(box_bow_coord_one.y)), radius, (73, 34, 124), thickness) # top-left
                    cv2.circle(image, (int(box_bow_coord_two.x), int(box_bow_coord_two.y)), radius, (73, 34, 124), thickness) # bottom - left
                    cv2.circle(image, (int(box_bow_coord_three.x), int(box_bow_coord_three.y)), radius, (73, 34, 124), thickness) # bottom - right
                    cv2.circle(image, (int(box_bow_coord_four.x), int(box_bow_coord_four.y)), radius, (73, 34, 124), thickness) # top - right

                    bow_coord_list.append(box_bow_coord_one)
                    bow_coord_list.append(box_bow_coord_two)
                    bow_coord_list.append(box_bow_coord_three)
                    bow_coord_list.append(box_bow_coord_four)

                    # Prepare text for box one
                    text_coord1 = f"Coord 1: ({box_bow_coord_one.x}, {box_bow_coord_one.y})"
                    text_coord2 = f"Coord 2: ({box_bow_coord_two.x}, {box_bow_coord_two.y})"
                    text_coord3 = f"Coord 3: ({box_bow_coord_three.x}, {box_bow_coord_three.y})"
                    text_coord4 = f"Coord 4: ({box_bow_coord_four.x}, {box_bow_coord_four.y})"

                    text_offset = 35  # increased spacing between lines
                    top_right_corner_text_two = (image.shape[1] - 370, text_offset + 20) # Adjusted to move down and left
                    top_right_corner_coord1_2 = (image.shape[1] - 370, text_offset * 2 + 15) # Adjusted to move down and left
                    top_right_corner_coord2_2 = (image.shape[1] - 370, text_offset * 3 + 10) # Adjusted to move down and left
                    top_right_corner_coord3_2 = (image.shape[1] - 370, text_offset * 4 + 5) # Adjusted to move down and left
                    top_right_corner_coord4_2 = (image.shape[1] - 370, text_offset * 5 + 0) # Adjusted to move down and left

                    # Put text on image for box two
                    text_two = "Bow OBB Coords:"
                    cv2.putText(image, text_two, top_right_corner_text_two, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(image, text_coord1, top_right_corner_coord1_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(image, text_coord2, top_right_corner_coord2_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(image, text_coord3, top_right_corner_coord3_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(image, text_coord4, top_right_corner_coord4_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size

                    # Detect if bow too high or low
                    bow_too_high = (image.shape[1] - 370, text_offset * 11 + 0) # Adjusted to move down and left
                    if(len(bow_coord_list) == 4 and len(string_coord_list) == 4):
            
                        P1 = Point2D.find_point_p1(bow_coord_list[0], bow_coord_list[1]) # left mid point
                        P2 = Point2D.find_point_p1(bow_coord_list[2], bow_coord_list[3]) # right mid point
                        int1 = Point2D.find_intersection(P1,P2,box_str_point_one,box_str_point_three)
                        int2 = Point2D.find_intersection(P1,P2,box_str_point_two,box_str_point_four)
                        if Point2D.is_above_or_below(int1, box_str_point_three, box_str_point_four) or Point2D.is_above_or_below(int2, box_str_point_three, box_str_point_four):
                            cv2.putText(image, "Bow Too High", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 4)  # Reduced font size
                        else:
                            cv2.putText(image, "Bow Correctly placed", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 4)  # Reduced font size

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

            out.write(resized_frame)
            #cv2.imshow('MediaPipe Hands', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        #print(Path(__file__).parent.parent)
        path = str(Path(__file__).parent / "demo.avi")
        print(path)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            raise Exception("Failed to create output video file or file is empty")
        

    testList = os.path.join(hand_pose_dir, "frontend_refactor\\Screenshot 2025-02-17 210532.png")
    print(testList)

def processFrame(image):
    # YOLOv8 model trained from Roboflow dataset
    # Used for bow and target area oriented bounding boxes
    
    hand_pose_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    model_dir = os.path.join(hand_pose_dir, "bow_target.pt")
    model = YOLO(model_dir)
    
    # For webcam input:
    # model.overlap = 80

    # dir = os.path.dirname(os.path.abspath(__file__))
    bkgd_path = os.path.join("", 'images\white_background.jpg')
    background = cv2.imread(bkgd_path)

    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    background = np.ones((height, width, channels), dtype = np.uint8) * 255

    #height,width = 480, 640
    #transparent_img = np.zeros((height, width, 4), dtype=np.uint8)

    #input video file
    #video_file_path = 'C:\\Users\\pelor\\Documents\\Coding\\Evaluator-code\\src\\computer_vision\\hand_pose_detection\\Too much pronation (1).mp4'
    #'/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/bow placing too high.mp4'
    

    os.makedirs('images', exist_ok=True)

    # Generate the path where the image will be saved
    image_path = os.path.join('', '/images/img.jpg')  # You can change the file extension if needed

    # Save the image to the specified path
    cv2.imwrite(image_path, image)

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
        base_options=BaseOptions(model_asset_buffer=open(gesture_model, "rb").read()),
        running_mode=VisionRunningMode.VIDEO,
        num_hands = num_hands)
  
    num_none = 0
    num_supination = 0
    num_correct = 0
    display_gesture = "none"

    desired_fps = 30 
    frame_delay = int(1000 / desired_fps)

    #set up hands and body
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6) as pose, GestureRecognizer.create_from_options(gesture_options) as recognizer:

    
        writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(output_frame_length,output_frame_width)) # algo makes a frame every ~80ms = 12.5 fps
       # while cap.isOpened():
        #    success, image = cap.read()
         #   if not success:
          #      break
  
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
        image = cv2.flip(image, 1) #FLIP ON Y
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

        if max(num_correct, num_none, num_supination) == num_supination:
            display_gesture = "supination"
        elif max(num_correct, num_none, num_supination) == num_correct:
            display_gesture = "correct"
        else:
            display_gesture = "none"
        num_none = 0
        num_supination = 0
        num_correct = 0
            
            # obtain neccesary data into array for display (using array because there are two hands)
        if gesture_recognition_result is not None and any(gesture_recognition_result.gestures):
            for single_hand_gesture_data in gesture_recognition_result.gestures:
                gesture_name = single_hand_gesture_data[0].category_name
                current_gestures.append(gesture_name)

            for single_hand_handedness_data in gesture_recognition_result.handedness:
                hand_name = single_hand_handedness_data[0].category_name
                current_handedness.append(hand_name)

            for single_hand_score_data in gesture_recognition_result.gestures:
                score = single_hand_score_data[0].score
                current_score.append(round(score, 2))

        y_pos = image.shape[0] - 70
        for x in range(len(current_gestures)):
            if current_handedness[x] != "Left":
                # increment number of none/supination for past 10 frames
                if current_gestures[x] == "supination":
                    num_supination += 1
                elif current_gestures[x] == "correct":
                    num_correct += 1
                else:
                    num_none += 1
            
                    # display classified gesture data on frames ->need classifaction for later, remove placement on image
                txt = current_handedness[x] + ": " + display_gesture + " " + str(current_score[x])
                if (display_gesture == "supination"):
                    cv2.putText(image, txt, (image.shape[1] - 600, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (218,10,3), 4, cv2.LINE_AA)
                    print(txt)
                else:
                    cv2.putText(image, txt, (image.shape[1] - 650, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 2, (37,245,252), 4, cv2.LINE_AA)
                    print(txt)

        bow_coord_list = []
        string_coord_list =[]
        YOLOresults = model(image)
        newList = []
        for result in YOLOresults:
            if len(result.obb.xyxyxyxy) > 0:
                coord_box_one = result.obb.xyxyxyxy[0]
                round_coord_box_one = torch.round(coord_box_one)

                box_str_coordinate_1 = round_coord_box_one[0]  # First coordinate (x1, y1)
                box_str_coordinate_2 = round_coord_box_one[1]   # Second coordinate (x2, y2)
                box_str_coordinate_3 = round_coord_box_one[2]   # Third coordinate (x3, y3)
                box_str_coordinate_4 = round_coord_box_one[3]   # Fourth coordinate (x4, y4)

                    #create Point2D objects for each box coordinate
                box_str_point_one = Point2D(box_str_coordinate_1[0].item(), box_str_coordinate_1[1].item())
                box_str_point_two = Point2D(box_str_coordinate_2[0].item(), box_str_coordinate_2[1].item())
                box_str_point_three = Point2D(box_str_coordinate_3[0].item(), box_str_coordinate_3[1].item())
                box_str_point_four = Point2D(box_str_coordinate_4[0].item(), box_str_coordinate_4[1].item())

                newList.append(("box string top left", box_str_point_one))
                newList.append(("box string top right", box_str_point_two))
                newList.append(("box string bottom right", box_str_point_three))
                newList.append(("box string bottom left", box_str_point_four))

                    # Prepare text
                """
                text_one = "String OBB Coords:"
                text_coord1 = f"Coord 1: ({box_str_point_one.x}, {box_str_point_one.y})"
                text_coord2 = f"Coord 2: ({box_str_point_two.x}, {box_str_point_two.y})"
                text_coord3 = f"Coord 3: ({box_str_point_three.x}, {box_str_point_three.y})"
                text_coord4 = f"Coord 4: ({box_str_point_four.x}, {box_str_point_four.y})"
                """

                    # Define the color and size of the dot
                radius = 5           # Radius of the dot
                thickness = -1       # Thickness -1 fills the circle, creating a dot

                    #SHOWING DOTS
                cv2.circle(background, (int(box_str_point_one.x), int(box_str_point_one.y)), radius, (255, 0, 0), thickness) # bottom left
                cv2.circle(background, (int(box_str_point_two.x), int(box_str_point_two.y)), radius, (0, 0, 0), thickness) # bottom right
                cv2.circle(background, (int(box_str_point_three.x), int(box_str_point_three.y)), radius, (0, 255, 0), thickness) # top right                    
                cv2.circle(background, (int(box_str_point_four.x), int(box_str_point_four.y)), radius, (0, 0, 255), thickness) # top left
                string_coord_list.append(box_str_point_one)
                string_coord_list.append(box_str_point_two)
                string_coord_list.append(box_str_point_three)
                string_coord_list.append(box_str_point_four)

                
                    # Define bottom left corners for each text line
                bottom_left_corner_text_one = (image.shape[1] - 370, 35 * 6 + 20)  # Adjusted to move higher
                bottom_left_corner_coord1 = (image.shape[1] - 370, 35 * 7 + 15)   # Adjusted to move higher
                bottom_left_corner_coord2 = (image.shape[1] - 370, 35 * 8 + 10)    # Adjusted to move higher
                bottom_left_corner_coord3 = (image.shape[1] - 370, 35 * 9 + 5)    # Adjusted to move higher
                bottom_left_corner_coord4 = (image.shape[1] - 370, 35 * 10 + 0)    # Adjusted to move higher
                    # Put text on image for box one
                """
                cv2.putText(image, text_one, bottom_left_corner_text_one, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                cv2.putText(image, text_coord1, bottom_left_corner_coord1, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                cv2.putText(image, text_coord2, bottom_left_corner_coord2, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                cv2.putText(image, text_coord3, bottom_left_corner_coord3, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                cv2.putText(image, text_coord4, bottom_left_corner_coord4, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                """
                    # CALCULATING P1
                    #pointOne = Point2D.find_point_p1(leftCoordOne, rightCoordTwo, ratio=0.7)


            if len(result.obb.xyxyxyxy) >= 2:
                coord_box_two = result.obb.xyxyxyxy[1]
                round_coord_box_two = torch.round(coord_box_two)

                box_bow_coordinate_1 = round_coord_box_two[0]  # First coordinate (x1, y1)
                box_bow_coordinate_2 = round_coord_box_two[1]  # Second coordinate (x2, y2)
                box_bow_coordinate_3 = round_coord_box_two[2]  # Third coordinate (x3, y3)
                box_bow_coordinate_4 = round_coord_box_two[3]  # Fourth coordinate (x4, y4)

                    # Define the color and size of the dot
                radius = 5           # Radius of the dot
                thickness = -1       # Thickness -1 fills the circle, creating a dot
                    # Add the dot to the image at the specified coordinates
                box_bow_coord_one = Point2D(box_bow_coordinate_1[0].item(), box_bow_coordinate_1[1].item())
                box_bow_coord_two = Point2D(box_bow_coordinate_2[0].item(), box_bow_coordinate_2[1].item())
                box_bow_coord_three = Point2D(box_bow_coordinate_3[0].item(), box_bow_coordinate_3[1].item())
                box_bow_coord_four = Point2D(box_bow_coordinate_4[0].item(), box_bow_coordinate_4[1].item())
                    # SHOWING DOTS
                cv2.circle(background, (int(box_bow_coord_one.x), int(box_bow_coord_one.y)), radius, (73, 34, 124), thickness) # top-left
                cv2.circle(background, (int(box_bow_coord_two.x), int(box_bow_coord_two.y)), radius, (73, 34, 124), thickness) # bottom - left
                cv2.circle(background, (int(box_bow_coord_three.x), int(box_bow_coord_three.y)), radius, (73, 34, 124), thickness) # bottom - right
                cv2.circle(background, (int(box_bow_coord_four.x), int(box_bow_coord_four.y)), radius, (73, 34, 124), thickness) # top - right

                bow_coord_list.append(box_bow_coord_one)
                bow_coord_list.append(box_bow_coord_two)
                bow_coord_list.append(box_bow_coord_three)
                bow_coord_list.append(box_bow_coord_four)

                sorted_points = sorted(bow_coord_list, key=lambda p: (p.y, p.x))
                print(sorted_points)
                newList.append(("box bow top left", sorted_points[2]))
                newList.append(("box bow top right", sorted_points[3]))
                newList.append(("box bow bottom left", sorted_points[0]))
                newList.append(("box bow bottom right", sorted_points[1]))

                """
                    # Prepare text for box one
                text_coord1 = f"Coord 1: ({box_bow_coord_one.x}, {box_bow_coord_one.y})"
                text_coord2 = f"Coord 2: ({box_bow_coord_two.x}, {box_bow_coord_two.y})"
                text_coord3 = f"Coord 3: ({box_bow_coord_three.x}, {box_bow_coord_three.y})"
                text_coord4 = f"Coord 4: ({box_bow_coord_four.x}, {box_bow_coord_four.y})"

                text_offset = 35  # increased spacing between lines
                top_right_corner_text_two = (image.shape[1] - 370, text_offset + 20) # Adjusted to move down and left
                top_right_corner_coord1_2 = (image.shape[1] - 370, text_offset * 2 + 15) # Adjusted to move down and left
                top_right_corner_coord2_2 = (image.shape[1] - 370, text_offset * 3 + 10) # Adjusted to move down and left
                top_right_corner_coord3_2 = (image.shape[1] - 370, text_offset * 4 + 5) # Adjusted to move down and left
                top_right_corner_coord4_2 = (image.shape[1] - 370, text_offset * 5 + 0) # Adjusted to move down and left
                """
                    # Put text on image for box two
                """
                text_two = "Bow OBB Coords:"
                cv2.putText(image, text_two, top_right_corner_text_two, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                cv2.putText(image, text_coord1, top_right_corner_coord1_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                cv2.putText(image, text_coord2, top_right_corner_coord2_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                cv2.putText(image, text_coord3, top_right_corner_coord3_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                cv2.putText(image, text_coord4, top_right_corner_coord4_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                """

                    # Detect if bow too high or low -> keep for classifaction, remove putting text later
                bow_too_high = (image.shape[1] - 370, 0 * 11 + 0) # Adjusted to move down and left
                if(len(bow_coord_list) == 4 and len(string_coord_list) == 4):
            
                    P1 = Point2D.find_point_p1(bow_coord_list[0], bow_coord_list[1]) # left mid point
                    P2 = Point2D.find_point_p1(bow_coord_list[2], bow_coord_list[3]) # right mid point
                    int1 = Point2D.find_intersection(P1,P2,box_str_point_one,box_str_point_three)
                    int2 = Point2D.find_intersection(P1,P2,box_str_point_two,box_str_point_four)
                    if Point2D.is_above_or_below(int1, box_str_point_three, box_str_point_four) or Point2D.is_above_or_below(int2, box_str_point_three, box_str_point_four):
                        cv2.putText(background, "Bow Too High", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 4)  # Reduced font size
                    else:
                        cv2.putText(background, "Bow Correctly placed", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 4)  # Reduced font size

        detections = sv.Detections.from_ultralytics(YOLOresults[0])

        background.flags.writeable = True
        background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = background.shape

        print(results)
        if results.multi_hand_landmarks: #NEED for finger node coords later
            print("Processing hands: ")
            for hand_landmarks in results.multi_hand_landmarks:
                for ids, landmrk in enumerate(hand_landmarks.landmark):
                    cx, cy = landmrk.x * image_width, landmrk.y * image_height
                    store_finger_node_coords(ids, cx, cy, finger_coords)
                    #code for transmitting finger coords
                    finger_point = Point2D(cx, cy)
                    finger_label = "hand_pt_" + str(ids)
                    newList.append((finger_label, finger_point))
                    #print(newList[-1])
                
                mp_drawing.draw_landmarks(
                    background,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                    landmark=pose_results.pose_landmarks.landmark[11:15]

                )
                
                mp_drawing.draw_landmarks(
                    background,
                    landmark_subset,
                    None,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=10, circle_radius=6))
        else:
            print("Not Processing hands: ")
                

        oriented_box_annotator = sv.OrientedBoxAnnotator()
        
        annotated_frame = oriented_box_annotator.annotate(
            scene=background,
            detections=detections
        )
        

        background = ResizeWithAspectRatio(background, height=800)
        """ 
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
        """

        resized_frame = cv2.resize(background, (output_frame_length, output_frame_width))

        output_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2BGRA)
        output_img[:, :, 3] = 0 
        white_thresh = 250 
        mask = ~((resized_frame[:, :, 0] > white_thresh) &
         (resized_frame[:, :, 1] > white_thresh) &
         (resized_frame[:, :, 2] > white_thresh))

        output_img[mask] = np.concatenate(
            (resized_frame[mask], np.full((np.count_nonzero(mask), 1), 255, dtype=np.uint8)),
            axis=1
        )


        bkgd_path = os.path.join("", 'images\CLEAR.png')

        cv2.imwrite(bkgd_path, output_img)

        bkgd_path = os.path.join("", 'images\imgOut.jpg')

        cv2.imwrite(bkgd_path, background)

        #cv2.imshow('MediaPipe Hands', image)

        #writer.release()
        #cv2.destroyAllWindows()

        #newList = bow_coord_list + string_coord_list
        #print("*****************************")
        #print(finger_coords)
        #print(newList)

        #adding supination/correct
        newList.append(("supination", display_gesture))
        
        print("List len:", len(newList))
        for item in newList:
            print(item)
        return newList

if __name__ == "__main__":
    main()

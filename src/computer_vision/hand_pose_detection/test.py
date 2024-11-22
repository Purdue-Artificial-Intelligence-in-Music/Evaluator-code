import cv2
import mediapipe as mp
import numpy as np
import csv

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import copy
import itertools

import os
import sys
import supervision as sv
import ultralytics
from ultralytics import YOLO
import torch

from datetime import datetime

base_directory = os.path.dirname(__file__)

# Calculate the correct path to the model directory
model_directory = os.path.join(base_directory, 'model', 'keypoint_classifier')
model_file = os.path.join(model_directory, 'keypoint_classifier.tflite')

# Ensure the model file exists
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

# Add the model_directory to sys.path
if model_directory not in sys.path:
    sys.path.append(model_directory)

from model import KeyPointClassifier

# Calculate the path to the 'computer_vision/utils' directory
utils_directory = os.path.join(base_directory, 'computer_vision', 'utils')

# Add the utils_directory to sys.path if it's not already included
if utils_directory not in sys.path:
    sys.path.append(utils_directory)

from utils import CvFpsCalc

# option setup for gesture recognizer
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

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

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    #
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def draw_info(image, fps, mode, number):
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, "FPS:" + str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv2.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv2.LINE_AA)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def logging_csv(number, mode, landmark_list, handedness):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        if handedness.classification[0].label[0:] == "Right":
            csv_path = 'src/computer_vision/hand_pose_detection/model/keypoint_classifier/keypoint.csv'
            with open(csv_path, 'a', newline="") as f:
                writer = csv.writer(f)
                writer.writerow([number, *landmark_list])
                print("Logged data")
    return


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def main():
    # YOLOv8 model trained from Roboflow dataset
    # Used for bow and target area oriented bounding boxes
    model = YOLO('/Users/felixlu/Desktop/Evaluator/Evaluator-code/src/computer_vision/hand_pose_detection/bow_target.pt')  # Path to your model file
  
    # For webcam input:
    # model.overlap = 80

    #input video file
    video_file_path = 'src/computer_vision/hand_pose_detection/Supination1.mp4'
    cap = cv2.VideoCapture(video_file_path) # change argument to 0 for demo/camera input

    frame_count = 0
    output_frame_length = 960
    output_frame_width = 720

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Initialize video writer
    output_file = 'output.mp4' 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # saves output video to output_file
    writer = cv2.VideoWriter(output_file, fourcc, 12.5, (output_frame_length, output_frame_width))

    #setup gesture options
    num_hands = 2

    # do majority vote
    num_none = 0
    num_supination = 0
    num_pronation = 0
    display_gesture = "none"
  
    use_brect = True

    # instantiate gesture classifier
    keypoint_classifier = KeyPointClassifier(model_path=model_file)

    relative_csv_path = 'model/keypoint_classifier/keypoint_classifier_label.csv'
    csv_path = os.path.join(base_directory, relative_csv_path)

    with open(csv_path, encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    mode = 0

    #set up hands and body
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands, mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6) as pose:
    
        writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(output_frame_length,output_frame_width)) # algo makes a frame every ~80ms = 12.5 fps
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # Process Key (ESC: end) #################################################
            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break

            number, mode = select_mode(key, mode)
            
            # calculate FPS
            fps = cvFpsCalc.get()
  
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            # this is actually important for some reason, flipping the image -> better classification accuracy
            image = cv2.flip(image, 1)
            debug_image = copy.deepcopy(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # classify hand signs
            results = hands.process(image)
            # classify body pose
            pose_results = pose.process(image)

            frame_count += 1

            # display majority class every 15 frames
            if (frame_count % 15 == 0):
                if max(num_pronation, num_none, num_supination) == num_supination:
                    display_gesture = "Supination"
                elif max(num_pronation, num_none, num_supination) == num_pronation:
                    display_gesture = "Pronation"
                else:
                    display_gesture = "Correct!"
                num_none = 0
                num_supination = 0
                num_pronation = 0
            
            # Draw hand landmarks
            if results.multi_hand_landmarks is not None:

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)

                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                     # Write to the dataset file
                    logging_csv(number, mode, pre_processed_landmark_list, handedness)

                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if (keypoint_classifier_labels[hand_sign_id] == "Pronation"):
                        num_none += 1
                    elif (keypoint_classifier_labels[hand_sign_id] == "Supination"):
                        num_supination += 1
                    else:
                        num_none += 1

                    # Draw hands
                    debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        display_gesture
                    )

                    # Draw pose
                    landmark_subset = landmark_pb2.NormalizedLandmarkList(
                        landmark=pose_results.pose_landmarks.landmark[11:15]
                    )

                    # circles settings
                    mp_drawing.draw_landmarks(
                        debug_image,
                        landmark_subset,
                        None,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=10, circle_radius=6)
                    )
                    mp_drawing.draw_landmarks(
                        debug_image,
                        landmark_subset,
                        None,
                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=10)
                    )
            # end if

            """
            bow_coord_list = []
            string_coord_list =[]
            YOLOresults = model(debug_image)
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
                    cv2.circle(debug_image, (int(box_str_point_one.x), int(box_str_point_one.y)), radius, (255, 0, 0), thickness) # bottom left
                    cv2.circle(debug_image, (int(box_str_point_two.x), int(box_str_point_two.y)), radius, (0, 0, 0), thickness) # bottom right
                    cv2.circle(debug_image, (int(box_str_point_three.x), int(box_str_point_three.y)), radius, (0, 255, 0), thickness) # top right
                    cv2.circle(debug_image, (int(box_str_point_four.x), int(box_str_point_four.y)), radius, (0, 0, 255), thickness) # top left
                    string_coord_list.append(box_str_point_one)
                    string_coord_list.append(box_str_point_two)
                    string_coord_list.append(box_str_point_three)
                    string_coord_list.append(box_str_point_four)
                    # Define bottom left corners for each text line
                    bottom_left_corner_text_one = (debug_image.shape[1] - 370, 35 * 6 + 20)  # Adjusted to move higher
                    bottom_left_corner_coord1 = (debug_image.shape[1] - 370, 35 * 7 + 15)   # Adjusted to move higher
                    bottom_left_corner_coord2 = (debug_image.shape[1] - 370, 35 * 8 + 10)    # Adjusted to move higher
                    bottom_left_corner_coord3 = (debug_image.shape[1] - 370, 35 * 9 + 5)    # Adjusted to move higher
                    bottom_left_corner_coord4 = (debug_image.shape[1] - 370, 35 * 10 + 0)    # Adjusted to move higher
                    # Put text on image for box one
                    cv2.putText(debug_image, text_one, bottom_left_corner_text_one, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(debug_image, text_coord1, bottom_left_corner_coord1, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(debug_image, text_coord2, bottom_left_corner_coord2, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(debug_image, text_coord3, bottom_left_corner_coord3, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
                    cv2.putText(debug_image, text_coord4, bottom_left_corner_coord4, cv2.FONT_HERSHEY_SIMPLEX, .8, (167, 52, 53), 2)
            
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
                    top_right_corner_text_two = (debug_image.shape[1] - 370, text_offset + 20) # Adjusted to move down and left
                    top_right_corner_coord1_2 = (debug_image.shape[1] - 370, text_offset * 2 + 15) # Adjusted to move down and left
                    top_right_corner_coord2_2 = (debug_image.shape[1] - 370, text_offset * 3 + 10) # Adjusted to move down and left
                    top_right_corner_coord3_2 = (debug_image.shape[1] - 370, text_offset * 4 + 5) # Adjusted to move down and left
                    top_right_corner_coord4_2 = (debug_image.shape[1] - 370, text_offset * 5 + 0) # Adjusted to move down and left

                    # Put text on image for box two
                    text_two = "Bow OBB Coords:"
                    cv2.putText(debug_image, text_two, top_right_corner_text_two, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(debug_image, text_coord1, top_right_corner_coord1_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(debug_image, text_coord2, top_right_corner_coord2_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(debug_image, text_coord3, top_right_corner_coord3_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size
                    cv2.putText(debug_image, text_coord4, top_right_corner_coord4_2, cv2.FONT_HERSHEY_SIMPLEX, .8, (73, 34, 124), 2)  # Reduced font size

                    # Detect if bow too high or low
                    bow_too_high = (debug_image.shape[1] - 370, text_offset * 11 + 0) # Adjusted to move down and left
                    if(len(bow_coord_list) == 4 and len(string_coord_list) == 4):
            
                        P1 = Point2D.find_point_p1(bow_coord_list[0], bow_coord_list[1]) # left mid point
                        P2 = Point2D.find_point_p1(bow_coord_list[2], bow_coord_list[3]) # right mid point
                        int1 = Point2D.find_intersection(P1,P2,box_str_point_one,box_str_point_three)
                        int2 = Point2D.find_intersection(P1,P2,box_str_point_two,box_str_point_four)
                        if Point2D.is_above_or_below(int1, box_str_point_three, box_str_point_four) or Point2D.is_above_or_below(int2, box_str_point_three, box_str_point_four):
                            cv2.putText(debug_image, "Bow Too High", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 0, 0), 4)  # Reduced font size
                        else:
                            cv2.putText(debug_image, "Bow Correctly placed", bow_too_high, cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0), 4)  # Reduced font size

            detections = sv.Detections.from_ultralytics(YOLOresults[0])

            # draw bounding boxes
            oriented_box_annotator = sv.OrientedBoxAnnotator()
            annotated_frame = oriented_box_annotator.annotate(
                scene=debug_image,
                detections=detections
            )

            """

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            image = ResizeWithAspectRatio(image, height=800)
            debug_image = cv2.putText(debug_image, "Frame {}".format(frame_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                 1, (0, 0, 0), 4, cv2.LINE_AA)
            
            cv2.putText(debug_image, "Frame {}".format(frame_count), (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # Resize to specified output dimensions before writing
            resized_frame = cv2.resize(debug_image, (output_frame_length, output_frame_width))

            debug_image = draw_info(debug_image, fps, mode, number)

            writer.write(resized_frame)
            cv2.imshow('MediaPipe Hands', debug_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

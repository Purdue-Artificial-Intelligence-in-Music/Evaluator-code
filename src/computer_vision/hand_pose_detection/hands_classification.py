import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.framework.formats import landmark_pb2

import copy
import itertools

import os
import sys

base_directory = os.path.dirname(__file__)

# Calculate the correct path to the model directory
model_directory = os.path.join(base_directory, 'model', 'keypoint_classifier')
#model_file = os.path.join(model_directory, 'keypoint_classifier.tflite')

# Model paths for hand and elbow models 
model_file = 'keypoint_classifier_FINAL.tflite'
model_file_elbow = 'keypoint_classifier_shoulder.tflite'

# Ensure the model file exists
if not os.path.exists(model_file):
    raise FileNotFoundError(f"Model file not found: {model_file}")

# Add the model_directory to sys.path
if model_directory not in sys.path:
    sys.path.append(model_directory)

from model import KeyPointClassifier

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
    
#setup gesture options
num_hands = 2

finger_coords = {}

    
# instantiate gesture classifier
keypoint_classifier = KeyPointClassifier(model_path=model_file)
elbow_classifier = KeyPointClassifier(model_path=model_file_elbow)

class Hands: 

    # Function to resize image with aspect ratio
    def ResizeWithAspectRatio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
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
    def store_finger_node_coords(self, id: int, cx: float, cy: float, finger_coords: dict):
        if id not in finger_coords:
            finger_coords[id] = []
        finger_coords[id].append((cx, cy))

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv2.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def calc_landmark_list(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(self, landmark_list):
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
    
    def classify_elbow_posture(self, landmarks):

        shoulder = landmarks[11]
        elbow = landmarks[13]
        right_hand = landmarks[15]
        right_index = landmarks[19]

        shoulder_2d = np.array([shoulder.x, shoulder.y])
        elbow_2d = np.array([elbow.x, elbow.y])
        right_hand_2d = np.array([right_hand.x, right_hand.y])

        a_2d = shoulder_2d - elbow_2d
        b_2d = right_hand_2d - elbow_2d

        cos_theta_2d = np.dot(a_2d, b_2d) / (np.linalg.norm(a_2d) * np.linalg.norm(b_2d))
        theta_rad_2d = np.arccos(np.clip(cos_theta_2d, -1.0, 1.0))
        theta_deg_2d = np.degrees(theta_rad_2d)
 
        #finds angle at elbow using x, y, and z coordinates of shoulder, elbow, and hand
        shoulder_3d = np.array([shoulder.x, shoulder.y, shoulder.z])
        elbow_3d = np.array([elbow.x, elbow.y, elbow.z])
        right_hand_3d = np.array([right_hand.x, right_hand.y, right_hand.z])

        shoulder_elbow_dist_vec = shoulder_3d - elbow_3d
        hand_elbow_dist_vec = right_hand_3d - elbow_3d

        cos_theta_3d = np.dot(shoulder_elbow_dist_vec, hand_elbow_dist_vec) / (np.linalg.norm(shoulder_elbow_dist_vec)) * np.linalg.norm(hand_elbow_dist_vec)
        theta_rad_3d = np.arccos(np.clip(cos_theta_3d, -1.0, 1.0))
        theta_deg_3d = np.degrees(theta_rad_3d)

        shoulder_elbow_dist = np.linalg.norm(shoulder_elbow_dist_vec)
        hand_elbow_dist = np.linalg.norm(hand_elbow_dist_vec)

        shoulder_elbow_dist_norm = shoulder_elbow_dist_vec / shoulder_elbow_dist
        hand_elbow_dist_norm = hand_elbow_dist_vec / hand_elbow_dist

        return (shoulder_elbow_dist_norm[0], shoulder_elbow_dist_norm[1], shoulder_elbow_dist_norm[2], hand_elbow_dist_norm[0], hand_elbow_dist_norm[1], hand_elbow_dist_norm[2], theta_rad_3d, shoulder_elbow_dist, hand_elbow_dist)
    
    def process_frame(self, image, hands, pose):

        wrist_posture = 'None Detected'
        elbow_posture = 'None Detected'
        hand_coordinates = 'None Detected'

        with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands, mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.6) as pose:

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            image = cv2.flip(image, 1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # mp hand model
            results = hands.process(image)

            # mp pose model
            pose_results = pose.process(image)

            image_height, image_width, _ = image.shape
            
            # Draw hand landmarks
            if results.multi_hand_landmarks is not None:

                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    for ids, landmrk in enumerate(hand_landmarks.landmark):
                        cx, cy = landmrk.x * image_width, landmrk.y * image_height
                        self.store_finger_node_coords(ids, cx, cy, finger_coords)
                            
                    # Check if the hand is the bow hand
                    if handedness.classification[0].label == 'Right':

                        hand_coordinates = hand_landmarks

                        # Landmark calculation
                        landmark_list = self.calc_landmark_list(image, hand_landmarks)

                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = self.pre_process_landmark(landmark_list)

                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                        if hand_sign_id == 0:
                            wrist_posture = 'Normal'
                        elif hand_sign_id == 1:
                            wrist_posture = 'Supination'
                        elif hand_sign_id == 2:
                            wrist_posture = 'Pronation'
                            
                    # Draw pose
                    if pose_results.pose_landmarks:
                                
                        landmark_subset = landmark_pb2.NormalizedLandmarkList(landmark=pose_results.pose_landmarks.landmark[11:15])
                            
                    if pose_results.pose_landmarks:

                        landmarks = pose_results.pose_landmarks.landmark

                        shoulder = landmarks[11]
                        elbow = landmarks[13]
                        wrist = landmarks[15]

                        shoulder_x = shoulder.x
                        shoulder_y = shoulder.y
                        elbow_x = elbow.x
                        elbow_y = elbow.y
                        wrist_x = wrist.x
                        wrist_y = wrist.y
                            
                    elbow_metrics = self.classify_elbow_posture(landmarks)

                    elbow_classification_id = elbow_classifier(elbow_metrics)

                    if elbow_classification_id == 0:
                        elbow_posture = 'Normal'
                    elif elbow_classification_id == 1:
                        elbow_posture = 'Too Low'
                    elif elbow_classification_id == 2:
                        elbow_posture = 'Too High'

        #gets rid of the z coordinate and flips the x-coordinates
        if hand_coordinates != "None Detected":
            hand_coordinates = [(1 - lm.x, lm.y) for lm in hand_landmarks.landmark]
                
        return (wrist_posture, elbow_posture, hand_coordinates, shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y)

import os
import csv
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import statistics
import math
import time

import mediapipe as mp
import copy
import itertools

from model import KeyPointClassifier


print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

base_directory = os.path.dirname(os.path.abspath(__file__))


class Classification:
    MAX_QUEUE_SIZE = 60
    MAX_Y_DELTA_THRESHOLD = 3
    MAX_BOW_DIST_THRESHOLD = 5

    def __init__(self):
        yolo_path = os.path.join(base_directory, 'robust_yolo.pt')
        self.model = YOLO(yolo_path if os.path.exists(yolo_path) else 'robust_yolo.pt')

        self.yolo_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.yolo_infer_device = 0 if torch.cuda.is_available() else 'cpu'
        self.model.to(self.yolo_device)
        print("Model device:", next(self.model.model.parameters()).device)

        self.bow_points = None
        self.string_points = None
        self.y_locked = False
        self.FRAME_COUNTER = 0
        self.num_wait_frames = 11
        self.y_avg = [0, 0]
        self.bow_repeat = 0
        self.string_repeat = 0
        self.delta_queue = []
        self.inference_times = []

    def update_points(self, string_box_xyxyxyxy, bow_box_xyxyxyxy):
        self.bow_points = bow_box_xyxyxyxy
        if string_box_xyxyxyxy is not None:
            self.string_points = self.update_string_points(string_box_xyxyxyxy)
            self.last_valid_string = self.string_points
        else:
            if hasattr(self, 'last_valid_string'):
                self.string_points = self.last_valid_string

    def get_midline(self):
        def distance(pt1, pt2):
            return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

        d1 = distance(self.bow_points[0], self.bow_points[1])
        d2 = distance(self.bow_points[1], self.bow_points[2])
        d3 = distance(self.bow_points[2], self.bow_points[3])
        d4 = distance(self.bow_points[3], self.bow_points[0])
        l = [d1, d2, d3, d4]
        min_index = l.index(min(l))

        if min_index == 0:
            pair1 = self.bow_points[0], self.bow_points[1]
            pair2 = self.bow_points[2], self.bow_points[3]
        elif min_index == 1:
            pair1 = self.bow_points[1], self.bow_points[2]
            pair2 = self.bow_points[3], self.bow_points[0]
        elif min_index == 2:
            pair1 = self.bow_points[2], self.bow_points[3]
            pair2 = self.bow_points[0], self.bow_points[1]
        else:
            pair1 = self.bow_points[3], self.bow_points[0]
            pair2 = self.bow_points[1], self.bow_points[2]

        mid1 = (pair1[0][0] + pair1[1][0]) / 2, (pair1[0][1] + pair1[1][1]) / 2
        mid2 = (pair2[0][0] + pair2[1][0]) / 2, (pair2[0][1] + pair2[1][1]) / 2

        dy = mid1[1] - mid2[1]
        dx = mid1[0] - mid2[0]
        if dx == 0:
            return float('inf'), mid1[0]
        slope = dy / dx
        intercept = mid1[1] - slope * mid1[0]

        return slope, intercept

    def get_vertical_lines(self):
        topLeft = self.string_points[0]
        topRight = self.string_points[1]
        botRight = self.string_points[2]
        botLeft = self.string_points[3]

        dx_left = topLeft[0] - botLeft[0]
        if dx_left == 0:
            leftSlope = float('inf')
            leftYint = None
        else:
            leftSlope = (topLeft[1] - botLeft[1]) / dx_left
            leftYint = topLeft[1] - leftSlope * topLeft[0]

        dx_right = topRight[0] - botRight[0]
        if dx_right == 0:
            rightSlope = float('inf')
            rightYint = None
        else:
            rightSlope = (topRight[1] - botRight[1]) / dx_right
            rightYint = topRight[1] - rightSlope * topRight[0]

        leftHT1 = self.string_points[0][1]
        leftHT2 = self.string_points[3][1]
        rightHT1 = self.string_points[1][1]
        rightHT2 = self.string_points[2][1]

        return (leftSlope, leftYint, leftHT1, leftHT2), (rightSlope, rightYint, rightHT1, rightHT2)

    def intersects_vertical(self, linear_line, vertical_lines):
        m, b = linear_line
        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        def get_intersection(v_line, x_ref):
            slope_v, intercept_v, top_y, bot_y = v_line

            if slope_v == float("inf") or intercept_v is None:
                x = x_ref
                if m == float('inf'):
                    return None
                y = m * x + b
            elif m == float('inf'):
                x = b
                y = slope_v * x + intercept_v
            elif abs(m - slope_v) < 1e-6:
                return None
            else:
                x = (intercept_v - b) / (m - slope_v)
                y = m * x + b

            ymin = min(top_y, bot_y)
            ymax = max(top_y, bot_y)

            if not (ymin <= y <= ymax):
                return None

            return (x, y)

        x_left = self.string_points[0][0]
        x_right = self.string_points[1][0]

        pt1 = get_intersection(vertical_one, x_left)
        pt2 = get_intersection(vertical_two, x_right)

        if pt1 is None or pt2 is None:
            return 1

        return self.bow_height_intersection((pt1, pt2), vertical_lines)

    @staticmethod
    def sort_string_points(pts):
        sorted_pts = sorted(pts, key=lambda x: x[1])
        top_points = sorted_pts[:2]
        bottom_points = sorted_pts[2:]
        top_points = sorted(top_points, key=lambda x: x[0])
        bottom_points = sorted(bottom_points, key=lambda x: x[0], reverse=True)
        return np.array(top_points + bottom_points)

    def sort_bow_points(self, pts):
        sorted_pts = sorted(pts, key=lambda p: p[0])
        left_points = sorted(sorted_pts[:2], key=lambda p: p[1])
        right_points = sorted(sorted_pts[2:], key=lambda p: p[1])
        return np.array([left_points[0], right_points[0], right_points[1], left_points[1]])

    def update_string_points(self, string_box):
        sorted_string = self.sort_string_points(string_box)

        delta_y = abs(sorted_string[0][1] - sorted_string[3][1])
        self.delta_queue.append(delta_y)
        if len(self.delta_queue) > self.MAX_QUEUE_SIZE:
            self.delta_queue.pop(0)

        if len(self.delta_queue) == 1:
            self.string_points = sorted_string
            return sorted_string

        median_delta = statistics.median(self.delta_queue)

        if abs(median_delta - delta_y) > self.MAX_Y_DELTA_THRESHOLD:
            sorted_string[0][1] = sorted_string[3][1] - median_delta
            sorted_string[1][1] = sorted_string[2][1] - median_delta

        if self.bow_points is not None:
            sorted_bow = self.sort_bow_points(self.bow_points)
            top_avg_bow_y = (sorted_bow[0][1] + sorted_bow[1][1]) / 2
            top_avg_str_y = (sorted_string[0][1] + sorted_string[1][1]) / 2
            bot_avg_bow_y = (sorted_bow[2][1] + sorted_bow[3][1]) / 2

            if (top_avg_bow_y <= top_avg_str_y and
                bot_avg_bow_y >= (top_avg_str_y - self.MAX_BOW_DIST_THRESHOLD)):
                if (sorted_bow[0][0] < sorted_string[0][0] and
                    sorted_bow[1][0] > sorted_string[1][0]):
                    sorted_string[0][1] = sorted_string[3][1] - median_delta
                    sorted_string[1][1] = sorted_string[2][1] - median_delta

        return sorted_string

    def bow_height_intersection(self, intersection_points, vertical_lines):
        top_zone_percentage = 0.1
        bottom_zone_percentage = 0.1

        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        top_y1 = vertical_one[2]
        top_y2 = vertical_two[2]
        bot_y1 = vertical_one[3]
        bot_y2 = vertical_two[3]

        height = abs(((bot_y1 - top_y1) + (bot_y2 - top_y2)) / 2.0)
        if height == 0:
            return 0

        avg_top_y = (top_y1 + top_y2) / 2.0
        avg_bot_y = (bot_y1 + bot_y2) / 2.0

        too_high_threshold = avg_top_y + height * top_zone_percentage
        too_low_threshold = avg_bot_y - height * bottom_zone_percentage

        intersection_y = (intersection_points[0][1] + intersection_points[1][1]) / 2.0

        if intersection_y <= too_high_threshold:
            return 2

        if intersection_y >= too_low_threshold:
            return 3

        return 0

    def average_y_coordinates(self, string_box_xyxyxyxy):
        string_box_xyxyxyxy = self.sort_string_points(string_box_xyxyxyxy)
        self.frame_num += 1
        y_coords = [pt[1] for pt in string_box_xyxyxyxy]
        self.string_ycoord_heights.append(y_coords)
        if self.frame_num % self.num_wait_frames:
            top_left_avg = statistics.median([frame[0] for frame in self.string_ycoord_heights])
            top_right_avg = statistics.median([frame[1] for frame in self.string_ycoord_heights])
            self.string_points = string_box_xyxyxyxy
            self.string_points[0][1] = top_left_avg
            self.string_points[1][1] = top_right_avg
            self.y_avg = [top_left_avg, top_right_avg]
            self.y_locked = True
            self.string_ycoord_heights = []

    def bow_angle(self, bow_line, vertical_lines):
        max_angle = 20

        m_bow = bow_line[0]
        m1 = vertical_lines[0][0]
        m2 = vertical_lines[1][0]

        angle_one = abs(math.degrees(math.atan(abs(m_bow - m2) / (1 + m_bow * m2))))
        angle_two = abs(math.degrees(math.atan(abs(m1 - m_bow) / (1 + m1 * m_bow))))

        min_angle = min(abs(90 - min(angle_one, angle_two)), min(angle_one, angle_two))

        if min_angle > max_angle:
            return 1

        return 0

    def display_classification(self, result, opencv_frame):
        label_map = {
            0: ("Correct Bow Height", (0, 255, 0)),
            1: ("Outside Bow Zone", (0, 0, 255)),
            2: ("Too Low", (0, 165, 255)),
            3: ("Too High", (255, 0, 0))
        }

        if result in label_map:
            label, color = label_map[result]
        else:
            label = "Unknown"
            color = (255, 255, 255)

        cv2.putText(opencv_frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        for point in [self.bow_points, self.string_points]:
            if point is not None and point.any() and len(point) == 4:
                points = [tuple(map(int, p)) for p in point]
                cv2.polylines(opencv_frame, [np.array(points)], isClosed=True, color=color, thickness=2)

        return opencv_frame

    def print_timing_table(self):
        if not self.inference_times:
            print("No timing data collected.")
            return

        times_ms = [t * 1000 for t in self.inference_times[1:]] if len(self.inference_times) > 1 else [self.inference_times[0] * 1000]
        avg = statistics.mean(times_ms)
        med = statistics.median(times_ms)
        mn = min(times_ms)
        mx = max(times_ms)

        print("\n" + "=" * 55)
        print(f"{'INFERENCE TIMING SUMMARY':^55}")
        print("=" * 55)
        print(f"{'Metric':<25} {'Value':>20}")
        print("-" * 55)
        print(f"{'Total Frames':<25} {len(times_ms):>20}")
        print(f"{'Avg Inference (ms)':<25} {avg:>20.2f}")
        print(f"{'Median Inference (ms)':<25} {med:>20.2f}")
        print(f"{'Min Inference (ms)':<25} {mn:>20.2f}")
        print(f"{'Max Inference (ms)':<25} {mx:>20.2f}")
        print(f"{'Avg FPS':<25} {1000 / avg:>20.2f}")
        print("=" * 55)

    def process_frame(self, frame):
        return_dict = {"class": None, "bow": None, "string": None, "angle": None}

        t_start = time.perf_counter()
        results = self.model(frame, device=self.yolo_infer_device, verbose=False)
        t_end = time.perf_counter()
        self.inference_times.append(t_end - t_start)

        string_coords = None
        bow_coords = None
        if len(results) == 0:
            return None

        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and result.obb.xyxyxyxy is not None:
                if len(result.obb.xyxyxyxy) >= 1:
                    bow_conf = 0.0
                    bow_index = -1
                    string_conf = 0.0
                    string_index = -1

                    for x in range(len(result.obb)):
                        if int(result.obb.cls[x].item()) == 0:
                            if result.obb[x].conf > bow_conf:
                                bow_conf = result.obb[x].conf
                                bow_index = x
                        elif int(result.obb.cls[x].item()) == 1:
                            if result.obb[x].conf > string_conf:
                                string_conf = result.obb[x].conf
                                string_index = x

                    if bow_index != -1 and string_index != -1:
                        return_dict["bow"] = [tuple(torch.round(result.obb[bow_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                        return_dict["string"] = [tuple(torch.round(result.obb[string_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                        string_coords = self.sort_string_points(return_dict["string"])
                        bow_coords = np.array(return_dict["bow"])
                        self.string_repeat = 0
                        self.bow_repeat = 0
                    else:
                        return_dict["class"] = -1
                        if bow_index != -1:
                            self.bow_repeat = 0
                            return_dict["bow"] = [tuple(torch.round(result.obb[bow_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                            bow_coords = np.array(return_dict["bow"])
                            if self.string_repeat <= 6:
                                string_coords = self.string_points
                                self.string_repeat += 1
                            else:
                                string_coords = None
                        elif string_index != -1:
                            self.string_repeat = 0
                            return_dict["string"] = [tuple(torch.round(result.obb[string_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                            string_coords = self.sort_string_points(return_dict["string"])
                            if self.bow_repeat <= 6:
                                bow_coords = self.bow_points
                                self.bow_repeat += 1
                            else:
                                self.bow_repeat = 0
                else:
                    return_dict["class"] = -2
                    print("no detections")
                    return return_dict

                if string_coords is not None and bow_coords is not None:
                    self.update_points(string_coords, bow_coords)
                    midlines = self.get_midline()
                    vert_lines = self.get_vertical_lines()
                    intersect_points = self.intersects_vertical(midlines, vert_lines)
                    return_dict["angle"] = self.bow_angle(midlines, vert_lines)
                    return_dict["bow"] = [tuple(bow_coords[i].tolist()) for i in range(4)]
                    return_dict["string"] = [tuple(string_coords[i].tolist()) for i in range(4)]
                    return_dict["class"] = intersect_points
                else:
                    return_dict["class"] = -1

                return return_dict

        return return_dict


class Hands:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose

        self.max_num_hands = 2
        self.bow_hand_label = 'Right'
        self.pose_visibility_threshold = 0.5
        self.hand_wrist_selection_threshold = 0.1
        self.reference_ratio = 1.2
        self.elbow_threshold = 0.1

        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.6
        )

        hand_model_file = os.path.join(base_directory, '2_19_hands.tflite')
        elbow_model_file = os.path.join(base_directory, 'keypoint_classifier (1).tflite')
        label_file = os.path.join(base_directory, 'model', 'keypoint_classifier', 'keypoint_classifier_label.csv')

        self.keypoint_classifier = None
        self.elbow_classifier = None
        self.keypoint_classifier_labels = []

        if os.path.exists(hand_model_file):
            self.keypoint_classifier = KeyPointClassifier(model_path=hand_model_file)
        else:
            print('Warning: hand keypoint model not found. Hand classification disabled.')
            print(f'Expected model: {hand_model_file}')

        if os.path.exists(elbow_model_file):
            self.elbow_classifier = KeyPointClassifier(model_path=elbow_model_file)
        else:
            print('Warning: elbow keypoint model not found. Elbow classification disabled.')
            print(f'Expected model: {elbow_model_file}')

        if os.path.exists(label_file):
            with open(label_file, encoding='utf-8-sig') as f:
                self.keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
        else:
            print('Warning: label file not found. Using default labels.')
            self.keypoint_classifier_labels = ['Class 0', 'Class 1', 'Class 2']

    def close(self):
        self.hands.close()
        self.pose.close()

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

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

        max_value = max(list(map(abs, temp_landmark_list)))
        if max_value == 0:
            return temp_landmark_list

        def normalize_(n):
            return n / max_value

        temp_landmark_list = list(map(normalize_, temp_landmark_list))
        return temp_landmark_list

    def flip_landmark_list_x(landmark_list, image_width):
        flipped_landmark_list = copy.deepcopy(landmark_list)

        for landmark_point in flipped_landmark_list:
            landmark_point[0] = (image_width - 1) - landmark_point[0]

        return flipped_landmark_list

    def draw_bounding_rect(use_brect, image, brect):
        if use_brect:
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
        return image

    def draw_info_text(image, brect, handedness_label, hand_sign_text):
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

        info_text = handedness_label
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text

        cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    def draw_landmarks(image, landmark_point):
        if len(landmark_point) > 0:
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
            cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

            for index, landmark in enumerate(landmark_point):
                cv2.circle(image, tuple(landmark), 5, (255, 255, 255), -1)
                cv2.circle(image, tuple(landmark), 5, (0, 0, 0), 1)
                if index in [4, 8, 12, 16, 20]:
                    cv2.circle(image, tuple(landmark), 8, (255, 255, 255), -1)
                    cv2.circle(image, tuple(landmark), 8, (0, 0, 0), 1)

        return image

    def get_pose_triplet(self, pose_landmarks):
        if self.bow_hand_label == 'Right':
            shoulder_index = 12
            elbow_index = 14
            wrist_index = 16
        else:
            shoulder_index = 11
            elbow_index = 13
            wrist_index = 15

        shoulder = pose_landmarks[shoulder_index]
        elbow = pose_landmarks[elbow_index]
        wrist = pose_landmarks[wrist_index]

        if hasattr(shoulder, 'visibility') and shoulder.visibility < self.pose_visibility_threshold:
            return None, None, None
        if hasattr(elbow, 'visibility') and elbow.visibility < self.pose_visibility_threshold:
            return None, None, None
        if hasattr(wrist, 'visibility') and wrist.visibility < self.pose_visibility_threshold:
            return None, None, None

        return shoulder, elbow, wrist

    def build_elbow_feature_vector(self, shoulder, elbow, wrist):
        shoulder_x = 1.0 - shoulder.x
        shoulder_y = shoulder.y
        shoulder_z = shoulder.z

        elbow_x = 1.0 - elbow.x
        elbow_y = elbow.y
        elbow_z = elbow.z

        wrist_x = 1.0 - wrist.x
        wrist_y = wrist.y
        wrist_z = wrist.z

        shoulder_elbow_vec = np.array([
            shoulder_x - elbow_x,
            shoulder_y - elbow_y,
            shoulder_z - elbow_z,
        ], dtype=np.float32)

        wrist_elbow_vec = np.array([
            wrist_x - elbow_x,
            wrist_y - elbow_y,
            wrist_z - elbow_z,
        ], dtype=np.float32)

        shoulder_elbow_dist = float(np.linalg.norm(shoulder_elbow_vec))
        wrist_elbow_dist = float(np.linalg.norm(wrist_elbow_vec))

        features = np.zeros(9, dtype=np.float32)
        if shoulder_elbow_dist > 0 and wrist_elbow_dist > 0:
            shoulder_elbow_unit = shoulder_elbow_vec / shoulder_elbow_dist
            wrist_elbow_unit = wrist_elbow_vec / wrist_elbow_dist

            cos_theta = float(np.dot(shoulder_elbow_vec, wrist_elbow_vec) / (shoulder_elbow_dist * wrist_elbow_dist))
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta = float(math.acos(cos_theta))

            features[0:3] = shoulder_elbow_unit
            features[3:6] = wrist_elbow_unit
            features[6] = theta
            features[7] = shoulder_elbow_dist
            features[8] = wrist_elbow_dist

        return features.tolist()

    def select_hand_near_right_wrist(self, hand_results, pose_results):
        if not hand_results.multi_hand_landmarks or not hand_results.multi_handedness:
            return None

        if not pose_results.pose_landmarks:
            return None

        pose_landmarks = pose_results.pose_landmarks.landmark
        right_wrist_index = 16
        if len(pose_landmarks) <= right_wrist_index:
            return None

        pose_wrist = pose_landmarks[right_wrist_index]
        if hasattr(pose_wrist, 'visibility') and pose_wrist.visibility < self.pose_visibility_threshold:
            return None

        pose_wrist_x = pose_wrist.x
        pose_wrist_y = pose_wrist.y

        best_dist = float('inf')
        best_pair = None

        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            if not hand_landmarks.landmark:
                continue

            wrist = hand_landmarks.landmark[0]
            dx = wrist.x - pose_wrist_x
            dy = wrist.y - pose_wrist_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < self.hand_wrist_selection_threshold and dist < best_dist:
                best_dist = dist
                best_pair = (hand_landmarks, handedness)

        return best_pair

    def process_frame(self, frame):
        return_dict = {
            'hand_class': None,
            'handedness': None,
            'brect': None,
            'landmark_list': None,
            'elbow_class': None,
            'pose_points': None,
        }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)

        selected_pair = self.select_hand_near_right_wrist(hand_results, pose_results)
        if selected_pair is not None:
            hand_landmarks, handedness = selected_pair
            landmark_list = Hands.calc_landmark_list(frame, hand_landmarks)
            image_width = frame.shape[1]
            flipped_landmark_list = Hands.flip_landmark_list_x(landmark_list, image_width)
            pre_processed_landmark_list = Hands.pre_process_landmark(flipped_landmark_list)
            brect = Hands.calc_bounding_rect(frame, hand_landmarks)

            hand_sign_text = ''
            if self.keypoint_classifier is not None:
                hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                if 0 <= hand_sign_id < len(self.keypoint_classifier_labels):
                    hand_sign_text = self.keypoint_classifier_labels[hand_sign_id]
                else:
                    hand_sign_text = str(hand_sign_id)

            return_dict['hand_class'] = hand_sign_text
            return_dict['handedness'] = handedness.classification[0].label
            return_dict['brect'] = brect
            return_dict['landmark_list'] = landmark_list

        if pose_results.pose_landmarks:
            shoulder, elbow, wrist = self.get_pose_triplet(pose_results.pose_landmarks.landmark)
            if shoulder is not None and elbow is not None and wrist is not None:
                elbow_features = self.build_elbow_feature_vector(shoulder, elbow, wrist)

                elbow_class = 'Unknown'
                if self.elbow_classifier is not None:
                    elbow_sign_id = self.elbow_classifier(elbow_features)
                    elbow_classes = ['Normal Posture', 'Low Elbow', 'High Elbow']
                    if 0 <= elbow_sign_id < len(elbow_classes):
                        elbow_class = elbow_classes[elbow_sign_id]
                    else:
                        elbow_class = f'Class {elbow_sign_id}'

                return_dict['elbow_class'] = elbow_class
                return_dict['pose_points'] = [
                    (int(shoulder.x * frame.shape[1]), int(shoulder.y * frame.shape[0])),
                    (int(elbow.x * frame.shape[1]), int(elbow.y * frame.shape[0])),
                    (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                ]

        return return_dict

    def display_classification(self, result, opencv_frame):
        if result['brect'] is not None and result['landmark_list'] is not None:
            opencv_frame = Hands.draw_bounding_rect(True, opencv_frame, result['brect'])
            opencv_frame = Hands.draw_landmarks(opencv_frame, result['landmark_list'])
            opencv_frame = Hands.draw_info_text(
                opencv_frame,
                result['brect'],
                result['handedness'] if result['handedness'] is not None else self.bow_hand_label,
                result['hand_class'] if result['hand_class'] is not None else ''
            )

        if result['pose_points'] is not None:
            shoulder_point, elbow_point, wrist_point = result['pose_points']
            cv2.line(opencv_frame, shoulder_point, elbow_point, (255, 255, 255), 2)
            cv2.line(opencv_frame, elbow_point, wrist_point, (255, 255, 255), 2)
            cv2.circle(opencv_frame, shoulder_point, 6, (255, 255, 255), -1)
            cv2.circle(opencv_frame, elbow_point, 6, (255, 255, 255), -1)
            cv2.circle(opencv_frame, wrist_point, 6, (255, 255, 255), -1)

        if result['hand_class'] is not None:
            cv2.putText(opencv_frame, f"Hand: {result['hand_class']}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        if result['elbow_class'] is not None:
            elbow_color = (0, 255, 0) if result['elbow_class'] == 'Normal Posture' else (0, 0, 255)
            cv2.putText(opencv_frame, f"Elbow: {result['elbow_class']}", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, elbow_color, 2, cv2.LINE_AA)

        return opencv_frame


def main():
    input_video = os.path.join(base_directory, "bow too high-slow (3).mp4")
    output_video = os.path.join(base_directory, 'annotated_output2.mp4')

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_video}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    cln = Classification()
    hand_cln = Hands()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = frame.copy()
        c = cln.process_frame(annotated_frame)
        h = hand_cln.process_frame(annotated_frame)

        if c is not None and c["class"] is not None and c["class"] > -1:
            m, b = cln.get_midline()
            if m == float('inf'):
                x = int(b)
                cv2.line(annotated_frame, (x, 0), (x, annotated_frame.shape[0]), (0, 255, 255), 2)
            else:
                x0 = 0
                y0 = int(m * x0 + b)
                x1 = annotated_frame.shape[1]
                y1 = int(m * x1 + b)
                cv2.line(annotated_frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

            annotated_frame = cln.display_classification(c["class"], annotated_frame)

        annotated_frame = hand_cln.display_classification(h, annotated_frame)
        out.write(annotated_frame)

    cap.release()
    out.release()
    hand_cln.close()
    print(f"Video saved as '{output_video}'")
    cln.print_timing_table()


if __name__ == "__main__":
    main()

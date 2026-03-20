import cv2
import torch
import numpy as np
from ultralytics import YOLO
import statistics
import math
import time
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
class Classification:
    frame_num = 0
    string_ycoord_heights = []
    def __init__(self):
        self.model = YOLO('robust_yolo.pt')
        self.model.to('cuda')
        print("Model device:", next(self.model.model.parameters()).device)
        self.bow_points = None
        self.string_points = None
        self.y_locked = False
        self.FRAME_COUNTER = 0
        self.num_wait_frames = 11
        self.y_avg = [0, 0]
        self.bow_repeat = 0
        self.string_repeat = 0
        self.inference_times = []

    def update_points(self, string_box_xyxyxyxy, bow_box_xyxyxyxy):
        self.bow_points = bow_box_xyxyxyxy

        if string_box_xyxyxyxy is not None:
            if not self.y_locked:
                self.string_points = string_box_xyxyxyxy
            else:
                string_box_xyxyxyxy = self.sort_string_points(string_box_xyxyxyxy)
                self.string_points = string_box_xyxyxyxy
                self.string_points[0][1] = self.y_avg[0]
                self.string_points[1][1] = self.y_avg[1]
                print("y_avg:",self.y_avg)
                print(self.string_points)
            self.last_valid_string = self.string_points
        else:
            print("No new string box detected. Reusing last known string box.")
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
            0: ("Correct Bow Height", (0,255,0)),
            1: ("Outside Bow Zone", (0,0,255)),
            2: ("Too Low", (0,165,255)),
            3: ("Too High", (255,0,0))
        }

        if result in label_map:
            label, color = label_map[result]
        else:
            label = "Unknown"
            color = (255, 255, 255)

        cv2.putText(opencv_frame, label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
        for point in [self.bow_points, self.string_points]:
            if point.any() and len(point) == 4:
                points = [tuple(map(int,p)) for p in point]
                cv2.polylines(opencv_frame, [np.array(points)], isClosed=True, color=color, thickness=2)

        return opencv_frame

    def print_timing_table(self):
        if not self.inference_times:
            print("No timing data collected.")
            return

        times_ms = [t * 1000 for t in self.inference_times[1:]]  # skip first frame warmup
        avg = statistics.mean(times_ms)
        med = statistics.median(times_ms)
        mn = min(times_ms)
        mx = max(times_ms)

        print("\n" + "="*55)
        print(f"{'INFERENCE TIMING SUMMARY':^55}")
        print("="*55)
        print(f"{'Metric':<25} {'Value':>20}")
        print("-"*55)
        print(f"{'Total Frames':<25} {len(times_ms):>20}")
        print(f"{'Avg Inference (ms)':<25} {avg:>20.2f}")
        print(f"{'Median Inference (ms)':<25} {med:>20.2f}")
        print(f"{'Min Inference (ms)':<25} {mn:>20.2f}")
        print(f"{'Max Inference (ms)':<25} {mx:>20.2f}")
        print(f"{'Avg FPS':<25} {1000/avg:>20.2f}")
        print("="*55)

    def process_frame(self, frame):
        return_dict = {"class": None, "bow": None, "string": None, "angle": None}
        classes = ["bow", "string"]

        t_start = time.perf_counter()
        results = self.model(frame, device=0)
        t_end = time.perf_counter()
        self.inference_times.append(t_end - t_start)

        string_coords = None
        bow_coords = None
        if len(results) == 0:
            return None
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and result.obb.xyxyxyxy is not None:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy()
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
                    self.average_y_coordinates(string_coords)
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
                    else:
                        self.string_repeat = 0
                        return_dict["string"] = [tuple(torch.round(result.obb[string_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                        string_coords = self.sort_string_points(return_dict["string"])
                        self.average_y_coordinates(string_coords)
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
                result = intersect_points
                return_dict["bow"] = [tuple(bow_coords[i].tolist()) for i in range(4)]
                return_dict["string"] = [tuple(string_coords[i].tolist()) for i in range(4)]
                return_dict["class"] = result
            else:
                return_dict["class"] = -1
            return return_dict


def main():
    cap = cv2.VideoCapture("bow too high-slow (3).mp4")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('annotated_output2.mp4', fourcc, fps, (frame_width, frame_height))

    cln = Classification()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = frame.copy()
        c = cln.process_frame(annotated_frame)

        if c["class"] is not None and c["class"] > -1:
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

        out.write(annotated_frame)

    cap.release()
    out.release()
    print("Video saved as 'annotated_output2.mp4'")
    cln.print_timing_table()


if __name__ == "__main__":
    main()
import math
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import statistics

"""
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
        # Finds the coordinates of point P1 that is `ratio` distance from A to B.
        
        # Parameters:
        # A (Point2D): Point A
        # B (Point2D): Point B
        # ratio (float): Ratio of the distance from A to B where P1 should be (default is 0.7)
        
        # Returns:
        # Point2D: Coordinates of point P1
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
        # Determines if the current point (self) is above or below the line segment defined by points A and B.
        # Parameters:
        # A (Point2D): First endpoint of the line segment.
        # B (Point2D): Second endpoint of the line segment.
        # Returns:
        # bool: True if the current point (self) is above the line, False if it is below or on the line.
"""
        # Calculate the cross product of vectors AB and AC (where C is self)
        cross_product = (B.x - A.x) * (self.y - A.y) - (B.y - A.y) * (self.x - A.x)
        if cross_product > 0:
            return True  # Current point (self) is above the line
        else:
            return False 
    
    def is_above_or_below_list(self, pts_list):
        """
        # Determines if the current point (self) is above or below the line segment defined by points in pts_list.
        # Parameters:
        # pts_list (list of Point2D): List of points defining the line segment. Should be ordered from left to right.
        # Returns:
        # bool: True if the current point (self) is above the line, False if it is below or on the line.
"""
        for x in range(len(pts_list)):
            if x == 0:
                continue
            else:
                #If the self point is above the line segment formed by the current point and the previous point, then return True
                if self.is_above_or_below(pts_list[x], pts_list[x-1]):
                    return True
                #Otherwise do nothing and continue to the next point
        #If the self point is below all the line segments, then return False
        return False
    
    def check_correct_bow_position(bow_coord_list, string_coord_list):
        """
        # Determines if the bow is in the correct position.
        # Parameters:
        # bow_coord_list: list of Point2D, for the bow, from hand side to other side
        # string_coord_list: list of Point2D, for the cello strings, left to right for top strings then bottom strings
        # Does this by looking for a single intersection between the bow and a string
"""
        for x in range(len(string_coord_list)/2):
            if find_intersection(bow_coord_list[0], bow_coord_list[1], string_coord_list[x], string_coord_list[6+x]):
                return True
        return False

    @staticmethod
    def angle_between_lines(A, B, C, D):
        """
       # Calculates the angle between the line segment AB and CD.        
"""
        # Vectors AB and CD
        vector_ab = (B.x - A.x, B.y - A.y)
        vector_cd = (D.x - C.x, D.y - C.y)

        dot_product = vector_ab[0] * vector_cd[0] + vector_ab[1] * vector_cd[1]

        magnitude_ab = A.distance_to(B)
        magnitude_cd = C.distance_to(D)
      
       # Dot product property
        cos_theta = dot_product / (magnitude_ab * magnitude_cd)

        # Calculate arc cosine
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees

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

    def find_intersection(p1, p2, p3, p4):
        # Line 1: passing through p1 and p2
        A1 = p2[1] - p1[1]  # y2 - y1
        B1 = p1[0] - p2[0]  # x1 - x2
        C1 = A1 * p1[0] + B1 * p1[1]
    
        # Line 2: passing through p3 and p4
        A2 = p4[1] - p3[1]  # y4 - y3
        B2 = p3[0] - p4[0]  # x3 - x4
        C2 = A2 * p3[0] + B2 * p3[1]
    
        # Determinant of the system
        det = A1 * B2 - A2 * B1
    
        if det == 0:
            # Lines are parallel (no intersection)
            return None
        else:
            # Lines intersect, solving for x and y
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return (x, y)

    def find_intersection_list(self, p1, p2, pts_list):
        """
            # Determines Finds intersection between line segment p1p2 and line segments formed by points in pts_list.
            # Parameters:
            # p1: Point2D, first endpoint of the line segment
            # p2: Point2D, second endpoint of the line segment
            # pts_list: list of Point2D, for the cello strings, left to right for top strings then bottom strings
            # Returns:
            # bool: Coordinates of first intersection point found, if no intersection found then return None.
"""
        for x in range(len(pts_list)/2):
            intersection = find_intersection(p1, p2, pts_list[x], pts_list[6+x])
            if intersection:
                return intersection
        return None

    def check_angle(p1, p2, pts_list):
        """
        # Determines if there is a line segment that is almost parallel (within 10 degrees) to the line segment p1p2.
        # Parameters:
        # p1: Point2D, first endpoint of the line segment
        # p2: Point2D, second endpoint of the line segment
        # pts_list: list of Point2D, for the cello strings, left to right for top strings
"""
        for x in range(len(pts_list[:6])):
            if x == 0:
                continue
            else:
                angle = Point2D.angle_between_lines(p1, p2, pts_list[x], pts_list[x-1])
                if angle < 10:
                    return True
        return False
"""
class Classification:
    frame_num = 0
    string_ycoord_heights = []
    def __init__(self):
        self.bow_points = []  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples
        self.string_points = []  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples
        self.y_locked = False

    def update_points(self, string_box_xyxyxyxy, bow_box_xyxyxyxy):
        self.bow_points = bow_box_xyxyxyxy
        print('bow:', bow_box_xyxyxyxy)

        if string_box_xyxyxyxy is not None:
            if not self.y_locked:
                self.string_points = string_box_xyxyxyxy
            else:
                self.string_points = [
                    (new_x, old_y) for (new_x, _), (_, old_y) in zip(string_box_xyxyxyxy, self.string_points)
                ]
            self.last_valid_string = self.string_points  # save for fallback use
        else:
            print("No new string box detected. Reusing last known string box.")
            if hasattr(self, 'last_valid_string'):
                self.string_points = self.last_valid_string





    def get_midline(self):
        """
        Compute the midline (parallel to top and bottom edges) of the bow.
        Returns:
            - (m, b): slope and intercept for y = mx + b (midline),
                    OR (inf, x) for vertical line
        """
        botLeft, topLeft, topRight, botRight = self.bow_points

        # Get slope and intercept of top edge
        dx_top = topRight[0] - topLeft[0]
        dy_top = topRight[1] - topLeft[1]

        if dx_top == 0:
            # Perfect vertical bow
            mid_x = (topLeft[0] + botLeft[0]) / 2
            return float('inf'), mid_x

        slope = dy_top / dx_top
        intercept_top = topLeft[1] - slope * topLeft[0]

        # Get slope and intercept of bottom edge (should be nearly same slope)
        intercept_bottom = botLeft[1] - slope * botLeft[0]

        # Midline is average of intercepts
        intercept_mid = (intercept_top + intercept_bottom) / 2

        return slope, intercept_mid



    def get_vertical_lines(self):
        topLeft = self.string_points[0]
        topRight = self.string_points[1]
        botRight = self.string_points[2]
        botLeft = self.string_points[3]

        # Avoid division by zero by checking x-difference
        dx_left = topLeft[0] - botLeft[0]
        if dx_left == 0:
            leftSlope = float('inf')  # or use a large number like 1e6
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

        # [top-left, top-right, bottom-right, bottom-left]
        # (m1, b1, ht1, hb1)
        leftHT1 = self.string_points[0][1]
        leftHT2 = self.string_points[3][1]
        rightHT1 = self.string_points[1][1]
        rightHT2 = self.string_points[2][1]

        return (leftSlope, leftYint, leftHT1, leftHT2), (rightSlope, rightYint, rightHT1, rightHT2)

    def intersects_vertical(self, linear_line, vertical_lines):
        print('linear:', linear_line, '\nvertical:', vertical_lines)

        m, b = linear_line  # from get_midline()
        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        def get_intersection(v_line, x_ref):
            slope_v, intercept_v, top_y, bot_y = v_line

            if slope_v == float("inf") or intercept_v is None:
                # Vertical string line: x = constant
                x = x_ref
                if m == float('inf'):
                    return None  # both lines vertical
                y = m * x + b
            elif m == float('inf'):
                # Vertical bow midline
                x = b
                y = slope_v * x + intercept_v
            elif abs(m - slope_v) < 1e-6:
                return None  # Parallel lines
            else:
                x = (intercept_v - b) / (m - slope_v)
                y = m * x + b

            # Ensure top_y is *smaller* than bot_y (because image y increases downward)
            ymin = min(top_y, bot_y)
            ymax = max(top_y, bot_y)

            if not (ymin <= y <= ymax):
                print(f"Intersection y={y} is outside vertical range ({ymin}, {ymax})")
                return None

            return (x, y)


        x_left = self.string_points[0][0]  # top-left x
        x_right = self.string_points[1][0]  # top-right x

        pt1 = get_intersection(vertical_one, x_left)
        pt2 = get_intersection(vertical_two, x_right)

        if pt1 is None or pt2 is None:
            print("One or both intersections invalid")
            return 1

        return self.bow_height_intersection((pt1, pt2), vertical_lines)
    
    @staticmethod
    def sort_box_points_clockwise(pts):
        # Ensure numpy array for consistent indexing
        pts = np.array(pts)
        center = np.mean(pts, axis=0)

        def angle_from_center(pt):
            return np.arctan2(pt[1] - center[1], pt[0] - center[0])

        # Sort points clockwise around center
        sorted_pts = sorted(pts, key=angle_from_center)
        return pts


    def bow_height_intersection(self, intersection_points, vertical_lines):
        """
        Determines the height level at which the linear line intersects the vertical lines.

        Parameters:
        - intersection_points (tuple): ((x1, y1), (x2, y2)) coordinates of intersection for each line
        - vertical_lines (tuple): Output of get_vertical_lines() (m1, b1, ht1, hb1)

        Returns:
        - 3: Intersection is near top of the box (ht1 or ht2)
        - 2: Intersection is near bottom (hb1 or hb2)
        - 0: Intersection is in middle
        """
        print('intersection:', intersection_points)
        bot_scaling_factor = .25
        top_scaling_factor = .20

        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        bot_y1 = vertical_one[3]
        bot_y2 = vertical_two[3]
        
        top_y1 = vertical_one[2]
        top_y2 = vertical_two[2]

        # # get avg height of vertical lines
        # height = (top_y1 - bot_y1 + top_y2 - bot_y2 ) / 2

        # ## TOP AND BOTTOM CURRENTLY INTENTIONALLY FLIPPED

        # # get lower limit by averaging bottom y value. Scaled by height of strings and bot_scaling_factor
        # min = ((bot_y1 + bot_y2) / 2) + height * top_scaling_factor
        # print('min:', min)

        # if (intersection_points[0][1] <= min or intersection_points[1][1] <= min):
        #     return 3
        
        # # get upper limit by averaging top y value. Scaled by height of strings and bot_scaling_factor
        # max = ((top_y1 + top_y2) / 2) - height * bot_scaling_factor
        # print('max:', max)

        # if (intersection_points[0][1] >= max or intersection_points[1][1] >= max):
        #     return 2
        

                # get avg height of vertical lines
        height = ((bot_y1 - top_y1) + (bot_y2 - top_y2)) / 2

        min_y = ((top_y1 + top_y2) / 2) + height * top_scaling_factor
        print('min:', min_y)
        if (intersection_points[0][1] >= min_y or intersection_points[1][1] >= min_y):
            return 2

        max_y = ((bot_y1 + bot_y2) / 2) - height * bot_scaling_factor
        print('min:', max_y)
        if (intersection_points[0][1] <= max_y or intersection_points[1][1] <= max_y):
            return 3

        return 0

    def average_y_coordinates(self, string_box_xyxyxyxy):
        Classification.frame_num += 1
        y_coords = [pt[1] for pt in string_box_xyxyxyxy]
        Classification.string_ycoord_heights.append(y_coords)

        if Classification.frame_num == 61:
            top_y_avg = statistics.median([frame[0] for frame in Classification.string_ycoord_heights])  # top edge
            bot_y_avg = statistics.median([frame[3] for frame in Classification.string_ycoord_heights])  # bottom edge

            # Use original x-values, but lock Y-values at average top/bottom height
            self.string_points = [
                (string_box_xyxyxyxy[0][0], top_y_avg),  # top-left
                (string_box_xyxyxyxy[1][0], top_y_avg),  # top-right
                (string_box_xyxyxyxy[2][0], bot_y_avg),  # bottom-right
                (string_box_xyxyxyxy[3][0], bot_y_avg),  # bottom-left
            ]
            self.y_locked = True

    def bow_angle(self, bow_line, vertical_lines):
        """
        Classify the bow angle relative to the two vertical lines

        Parameters:
        - bow_line (tuple): (m, b)
        - vertical_lines (tuple): ((m1, b1, top1, bot1), (m2, b2, top2, bot2))

        Returns:
        - 1: Wrong Angle
        - 0: Correct Angle
        """
        max_angle = 20

        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        # Gets the angle converted to degrees using # arctan(|m1 - m2| / (1 + m1 * m2)). 
        # Uses the most-perpendicular of the two bow angles.
        angle_one = abs(math.degrees(math.atan(abs(bow_line[0] - vertical_two[0]) / (1 + bow_line[0] * vertical_two[0]))))
        angle_two = abs(math.degrees(math.atan(abs(vertical_one[0] - bow_line[0]) / (1 + vertical_one[0] * bow_line[0]))))
        
        print('angle:', max(angle_one, angle_two))
        
        if (max(angle_one, angle_two) < (90 - max_angle)):
            return 1
        
        return 0

    def display_classification(self, height_result, angle_result, opencv_frame):
        """
        Display a classification label based on the result code. Display bounding boxes and coordinates as well.

        Parameters:
        - result (int): Classification result code
            - 0: Correct bow height
            - 1: Outside bow zone
            - 2: Too low
            - 3: Too high

        Returns:
        - opencv frame
        """

        # define labels for height classification results
        height_label_map = {
            0: ("Correct Bow Height", (0,255,0)),
            1: ("Outside Bow Zone", (0,0,255)),
            2: ("Too Low", (0,165,255)),
            3: ("Too High", (255,0,0))
        }
        angle_label_map = {
            0: ("Correct Bow Angle", (0,255,0)),
            1: ("Improper Bow Angle", (0,0,255))
        }

        # define labels for angle classification results

        # if the result is in the label map, use that label and color
        if height_result in height_label_map:
            height_label, height_color = height_label_map[height_result]
        else:
            height_label = "Unknown"
            height_color = (255, 255, 255)
        
        if angle_result in angle_label_map:
            angle_label, angle_color = angle_label_map[angle_result]
        else:
            angle_label = "Unknown"
            angle_color = (255, 255, 255)

        cv2.putText(opencv_frame, height_label, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, height_color, 3, cv2.LINE_AA)
        #for point in [self.bow_points, self.string_points]:
        
        if self.string_points.any() and len(self.string_points) == 4:
                points = [tuple(map(int,p)) for p in self.string_points]
                cv2.polylines(opencv_frame, [np.array(points)], isClosed=True, color=height_color, thickness=2)

        if self.bow_points.any() and len(self.bow_points) == 4:
            points = [tuple(map(int,p)) for p in self.bow_points]
            cv2.polylines(opencv_frame, [np.array(points)], isClosed=True, color=angle_color, thickness=2)

        cv2.putText(opencv_frame, angle_label, (50,250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, angle_color, 3, cv2.LINE_AA)

        return opencv_frame
"""
    Main classification logic:
        within loop:
        1. Receive new bounding box corner coordinates.
        2. Call update_points() with the new coordinates.
        3. Compute and store vertical edge data using get_vertical_lines().
        4. Call intersects_vertical() with the current line and vertical edges.

        5. Interpret the return value from intersects_vertical():
            - 0 → Classify as "Correct"
            - 1 → Classify as "Outside Bow Zone"
            - 2 → Classify as "Too Low"
            - 3 → Classify as "Too High"

        6. Call display_classification(result, cv2_frame) to add classification labels.

        7. Display and show return value of display_classification.

        Note: if extra time check if computer uses gpu if not use this line:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
                if using gpu use this: 
                    obb_coords = result.obb.xyxyxyxy.cuda().numpy()  # shape: (N, 4, 2)
                    (might not need the.numpy() could be wrong though)

"""
def main():
    # Open video
    # Load YOLOv11 OBB model
    model = YOLO('/Users/jacksonshields/Documents/Evaluator/runs/obb/train4/weights/best.pt')  # Replace with your actual model file    
    cap = cv2.VideoCapture("/Users/jacksonshields/Downloads/right posture.mp4")
    # cap = cv2.VideoCapture("bow too high-slow (3).mp4")
    def resize_keep_aspect(image, target_width=1200):
        """Resize image while keeping aspect ratio"""
        h, w = image.shape[:2]
        scale = target_width / w
        new_dim = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    cln = Classification()
    FRAME_COUNTER = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv11 OBB inference
        annotated_frame = frame.copy()  # Initialize it safely with the original frame

        results = model(frame)
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and result.obb.xyxyxyxy is not None:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
                # for box in obb_coords:
                #     pts = box.reshape((-1, 1, 2)).astype(int)
                #     cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            FRAME_COUNTER += 1

            print("FRAME NUM ", FRAME_COUNTER)
            if len(result.obb.xyxyxyxy) >= 2:
                print("Both bow and string detected")
                if len(result.obb.xyxyxyxy) == 2:
                    if result.obb.cls[0] == result.obb.cls[1]:
                        continue #if both are bow or both are string, do nothing
                    if result.obb.cls[0] == 0: #first is bow, second is string
                        bow, string = torch.round(result.obb.xyxyxyxy)
                    else: #first is string, second is bow
                        string, bow = torch.round(result.obb.xyxyxyxy)
                else: #more than 2 detections, means there's probably a double detection of a bow or string
                    print("More than 2 detections")
                    bow_conf = 0.0
                    bow_index = -1
                    string_conf = 0.0
                    string_index = -1
                    for x in range(len(result.obb)):
                        if result.obb.cls[x] == 0:
                            if result.obb[x].conf > bow_conf:
                                bow_conf = result.obb[x].conf
                                bow_index = x
                        elif result.obb.cls[x] == 1:
                            if result.obb[x].conf > string_conf:
                                string_conf = result.obb[x].conf
                                string_index = x
                    if bow_index != -1 and string_index != -1:
                        bow = torch.round(result.obb[bow_index].xyxyxyxy)
                        string = torch.round(result.obb[string_index].xyxyxyxy)
                    else:
                        continue
                # bow_coords = [Point2D(bow[0][0].item(), bow[0][1].item()), Point2D(bow[1][0].item(), bow[1][1].item()), Point2D(bow[2][0].item(), bow[2][1].item()), Point2D(bow[3][0].item(), bow[3][1].item())]
                # string_coords = [Point2D(string[0][0].item(), string[0][1].item()), Point2D(string[1][0].item(), string[1][1].item()), Point2D(string[2][0].item(), string[2][1].item()), Point2D(string[3][0].item(), string[3][1].item())]
                
                # bow_coords = [tuple(bow[i].tolist()) for i in range(4)]
                # string_coords = [tuple(string[i].tolist()) for i in range(4)]
                if (len(bow) == 4 and len(string) == 4):
                    bow_coords = cln.sort_box_points_clockwise([tuple(bow[i].tolist()) for i in range(4)])
                    string_coords = cln.sort_box_points_clockwise([tuple(string[i].tolist()) for i in range(4)])

                if (FRAME_COUNTER <= 10):
                    cln.average_y_coordinates(string_coords)
                else:
                    if len(result.obb.xyxyxyxy) >= 2:
                        # You already handled this above
                        cln.update_points(string_coords, bow_coords)

                    elif len(result.obb.xyxyxyxy) == 1:
                        print("Only bow detected, reusing string coords from previous frame")
                        cln.update_points(cln.string_points, bow_coords)  # Use cached string x, y

                    # Draw bow midline
                    midline = cln.get_midline()
                    vert_lines = cln.get_vertical_lines()
                    intersect_points = cln.intersects_vertical(midline, vert_lines)
                    bow_angle = cln.bow_angle(midline, vert_lines)
                    
                    m, b = midline
                    if m == float('inf'):
                        x = int(b)
                        cv2.line(annotated_frame, (x, 0), (x, annotated_frame.shape[0]), (0, 255, 255), 2)
                    else:
                        x0 = 0
                        y0 = int(m * x0 + b)
                        x1 = annotated_frame.shape[1]
                        y1 = int(m * x1 + b)
                        cv2.line(annotated_frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

                    # Draw intersection points
                    if isinstance(intersect_points, tuple):
                        for pt in intersect_points:
                            cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1)
                    
                    annotated_frame = cln.display_classification(intersect_points, bow_angle, annotated_frame)

                

        


        # Resize frame for display
        resized_frame = resize_keep_aspect(annotated_frame, target_width=900)

        # resized_frame = resize_keep_aspect(frame, target_width=700)

        # Show the resized frame
        cv2.imshow("YOLOv11 OBB", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
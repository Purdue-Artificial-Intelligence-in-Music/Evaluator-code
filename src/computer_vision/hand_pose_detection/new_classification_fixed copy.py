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
        self.bow_points = None  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples
        self.string_points = None  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples
        self.y_locked = False
        self.FRAME_COUNTER = 0
        self.num_wait_frames = 11
        self.y_avg = [0, 0]

    def update_points(self, string_box_xyxyxyxy, bow_box_xyxyxyxy):
        self.bow_points = bow_box_xyxyxyxy
        print('String points:', self.string_points)

        if string_box_xyxyxyxy is not None:
            if not self.y_locked:
                self.string_points = string_box_xyxyxyxy
            else:
                """
                self.string_points = np.array([
                    (new_x, old_y) for (new_x, _), (_, old_y) in zip(string_box_xyxyxyxy, self.string_points)
                ])
                """
                self.string_points = string_box_xyxyxyxy
                self.string_points[0][1] = self.y_avg[0]
                self.string_points[1][1] = self.y_avg[1]
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

        def distance(pt1, pt2):
            return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

        d1 = distance(self.bow_points[0], self.bow_points[1])
        d2 = distance(self.bow_points[1], self.bow_points[2])
        d3 = distance(self.bow_points[2], self.bow_points[3])
        d4 = distance(self.bow_points[3], self.bow_points[0])
        l = [d1, d2, d3, d4]
        min_index = l.index(min(l))
        #two shortest distances are the two ends of the bow, use to get midpoints for midline
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
        if dy == 0:
            return float('inf'), mid1[0]
        slope = dy / dx
        intercept = mid1[1] - slope * mid1[0]
        
        return slope, intercept

        """topLeft, topRight, botRight, botLeft = pt1, pt2, pt3, pt4

        if self.bow_points is None:
            return None

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

        return slope, intercept_mid"""



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
        self.frame_num += 1
        y_coords = [pt[1] for pt in string_box_xyxyxyxy]
        self.string_ycoord_heights.append(y_coords)

        if self.frame_num == self.num_wait_frames:
            top_left_avg = statistics.median([frame[0] for frame in self.string_ycoord_heights])
            top_right_avg = statistics.median([frame[1] for frame in self.string_ycoord_heights])
            #top_y_avg = statistics.median([frame[0] for frame in self.string_ycoord_heights])  # top edge
            #bot_y_avg = statistics.median([frame[3] for frame in self.string_ycoord_heights])  # bottom edge

            # Use original x-values, but lock Y-values at average top height
            self.string_points = string_box_xyxyxyxy
            self.string_points[0][1] = top_left_avg
            self.string_points[1][1] = top_right_avg
            self.y_avg = [top_left_avg, top_right_avg]
            self.y_locked = True


    def display_classification(self, result, opencv_frame):
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

        # define labels for classification results
        label_map = {
            0: ("Correct Bow Height", (0,255,0)),
            1: ("Outside Bow Zone", (0,0,255)),
            2: ("Too Low", (0,165,255)),
            3: ("Too High", (255,0,0))
        }

        # if the result is in the label map, use that label and color
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
    
    def process_frame(self, frame):
        """
        returns dictionary with keys: class, bow, string
        class:
        -3: updating string y coordinates, not ready to classify
        -2: no detections
        -1: detection of only one object
        0: correct bow height
        1: outside bow zone
        2: too low
        3: too high
        
        
        """
        return_dict = {"class": None, "bow": None, "string": None}
        #expectation is that the frame is already resized to correct proportions
        classes = ["bow", "string"]
        model = YOLO('best.pt')  # Replace with your actual model file    
        results = model(frame)
        avg_frame_counter = False
        if len(results) == 0:
            return None
        #print("************", len(results[0].obb.xyxyxyxy), "************")
        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and result.obb.xyxyxyxy is not None:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
                # for box in obb_coords:
                #     pts = box.reshape((-1, 1, 2)).astype(int)
                #     cv2.polylines(annotated_frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
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
                    string_coords = np.array(return_dict["string"])
                    bow_coords = np.array(return_dict["bow"])
                else:
                    #only one object class detected
                    return_dict["class"] = -1
                    if bow_index != -1:
                        return_dict["bow"] = [tuple(torch.round(result.obb[bow_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                        bow_coords = np.array(return_dict["bow"])
                        string_coords = self.string_points
                    else:
                        return_dict["string"] = [tuple(torch.round(result.obb[string_index].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                        string_coords = np.array(return_dict["string"])
                        bow_coords = self.bow_points

                # Update the classification object with the detected coordinates
                #first check if there are any None values for bow or string points (means they havent been detected yet)
            else:
                return_dict["class"] = -2
                print("no detections")
                return return_dict
            if (return_dict["string"] is not None):
                if self.frame_num <= self.num_wait_frames:
                    self.average_y_coordinates(string_coords)
                    return_dict["class"] = -3
                    return return_dict
                
            
            if string_coords is not None and bow_coords is not None:
                self.update_points(string_coords, bow_coords)
                # Get bow midline and vertical lines
                midlines = self.get_midline()
                vert_lines = self.get_vertical_lines()
                intersect_points = self.intersects_vertical(midlines, vert_lines)

                # If vertical intersection returned 1 (invalid or out of bounds), classify as outside
                result = intersect_points  # Could be 0 (correct), 2 (too low), 3 (too high)
                return_dict["bow"] = [tuple(bow_coords[i].tolist()) for i in range(4)]
                return_dict["string"] = [tuple(string_coords[i].tolist()) for i in range(4)]
                return_dict["class"] = result
            else:
                return_dict["class"] = -1
            return return_dict
            """
            elif len(result.obb.xyxyxyxy) == 1:
                #print("Only one detection")
                #print(result.obb.cls[0], result.obb.xyxyxyxy)
                return_dict["class"] = -1
                return_dict[classes[int(result.obb.cls[0].item())]] = [tuple(torch.round(result.obb[0].xyxyxyxy)[0][i].tolist()) for i in range(4)]
                if int(result.obb.cls[0].item()) == 1 and self.FRAME_COUNTER <= num_wait_frames:
                    self.FRAME_COUNTER += 1
                    self.average_y_coordinates(return_dict["string"])
                    return_dict["class"] = -3
                return return_dict
            """
        
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
    model = YOLO('best.pt')  # Replace with your actual model file    
    cap = cv2.VideoCapture("Vertigo for Solo Cello - Cicely Parnas.mp4")
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

        c = cln.process_frame(annotated_frame)
        print(c)

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

            # Draw intersection points
#            if isinstance(cln.intersect_points, tuple):
#                for pt in cln.intersect_points:
#                    cv2.circle(annotated_frame, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1)
        
            annotated_frame = cln.display_classification(c["class"], annotated_frame)

                

        


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
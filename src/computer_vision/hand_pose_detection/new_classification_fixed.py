import cv2
import torch
import numpy as np
from ultralytics import YOLO

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
    
    def is_above_or_below_list(self, pts_list):
        """
        Determines if the current point (self) is above or below the line segment defined by points in pts_list.
        Parameters:
        pts_list (list of Point2D): List of points defining the line segment. Should be ordered from left to right.
        Returns:
        bool: True if the current point (self) is above the line, False if it is below or on the line.
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
        Determines if the bow is in the correct position.
        Parameters:
        bow_coord_list: list of Point2D, for the bow, from hand side to other side
        string_coord_list: list of Point2D, for the cello strings, left to right for top strings then bottom strings
        Does this by looking for a single intersection between the bow and a string
        """
        for x in range(len(string_coord_list)/2):
            if find_intersection(bow_coord_list[0], bow_coord_list[1], string_coord_list[x], string_coord_list[6+x]):
                return True
        return False

    @staticmethod
    def angle_between_lines(A, B, C, D):
        """
        Calculates the angle between the line segment AB and CD.        
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
            Determines Finds intersection between line segment p1p2 and line segments formed by points in pts_list.
            Parameters:
            p1: Point2D, first endpoint of the line segment
            p2: Point2D, second endpoint of the line segment
            pts_list: list of Point2D, for the cello strings, left to right for top strings then bottom strings
            Returns:
            bool: Coordinates of first intersection point found, if no intersection found then return None.
        """
        for x in range(len(pts_list)/2):
            intersection = find_intersection(p1, p2, pts_list[x], pts_list[6+x])
            if intersection:
                return intersection
        return None

    def check_angle(p1, p2, pts_list):
        """
        Determines if there is a line segment that is almost parallel (within 10 degrees) to the line segment p1p2.
        Parameters:
        p1: Point2D, first endpoint of the line segment
        p2: Point2D, second endpoint of the line segment
        pts_list: list of Point2D, for the cello strings, left to right for top strings
        """
        for x in range(len(pts_list[:6])):
            if x == 0:
                continue
            else:
                angle = Point2D.angle_between_lines(p1, p2, pts_list[x], pts_list[x-1])
                if angle < 10:
                    return True
        return False

class Classification:
    def __init__(self):
        self.bow_points = []  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples
        self.string_points = []  # Expected: [top-left, top-right, bottom-right, bottom-left] as (x, y) tuples

    def update_points(self, string_box_xyxyxyxy, bow_box_xyxyxyxy):
        """
        Update the internal box corner points.

        Parameters:
        - create two list of tuples, one for strings and one for bow:
		box_xyxyxyxy (list of tuple): List of 4 (x, y) tuples representing box corners in the order:
            [top-left, top-right, bottom-right, bottom-left]

        Returns:
        - None
        """
        self.bow_points = bow_box_xyxyxyxy
        self.string_points = string_box_xyxyxyxy


    def get_midline(self):
        """
        Compute the center horizontal midline between top and bottom edges.

        Returns:
        - (m, b): Slope and y-intercept of the midline in the form y = mx + b
        """

        topRight = self.bow_points[0]
        topLeft = self.bow_points[1]
        botRight = self.bow_points[0]
        botLeft = self.bow_points[1]

        topMid = ((topRight[0] + botRight[0]) / 2, (topRight[1] + botRight[1]) / 2)
        botMid = ((botLeft[0] + topLeft[0]) / 2, (botLeft[1] + topLeft[1]) / 2)
        
        slope = (topMid[1] - botMid[1]) / (topMid[0] - botMid[0])
        yInt = topMid[1] - slope * topMid[0]

        return (slope, yInt)

    def get_vertical_lines(self):
        """
        Compute the left and right vertical edges of the box.

        Returns:
        - (m1, b1, ht1, hb1): tuple of slope, x-intercept, top y, bottom y for left vertical line
        - (m2, b2, ht2, hb2): tuple of same for right vertical line
        """

        topRight = self.bow_points[0]
        topLeft = self.bow_points[1]
        botRight = self.bow_points[0]
        botLeft = self.bow_points[1]
        
        leftSlope = (topLeft[1] - botLeft[1]) / (topLeft[0] - botLeft[0])
        leftYint = topLeft[1] - leftSlope * topLeft[0]
        leftHT1 = topLeft[1]
        leftHT2 = botLeft[1]
 
        rightSlope = (topRight[1] - botRight[1]) / (topRight[0] - botRight[0])
        rightYint = topRight[1] - leftSlope * topRight[0]
        rightHT1 = topRight[1]
        rightHT2 = botRight[1]

        return (leftSlope, leftYint, leftHT1, leftHT2), (rightSlope, rightYint, rightHT1, rightHT2)
    def intersects_vertical(self, linear_line, vertical_lines):
        """
        Check if a linear line intersects both vertical edges of the box.

        Parameters:
        - linear_line (tuple): (m, b) for y = mx + b
        - vertical_lines (tuple): Output of get_vertical_lines()

        Returns:
        - If intersects both: calls bow_height_intersection(...)
        - If not: returns 1
        """
    
        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        # check if lines are parallel
        if (linear_line[0] == vertical_one[0] or linear_line[0] == vertical_two[0]):
            print('parallel')
            return 1

        # get coordinates of intersection for first vertical
        x_1 = (vertical_one[1] - linear_line[1]) / (linear_line[0] - vertical_one[0])
        y_1 = vertical_one[0] * x_1 + vertical_one[1]
        left = (vertical_one[3] - vertical_one[1]) / vertical_one[0]
        right = (vertical_one[2] - vertical_one[1]) / vertical_one[0]
        top = vertical_one[2]
        bot = vertical_one[3]

        # check coordinates are within target
        if ((bot > y_1 or top < y_1) or (left > x_1 or right < x_1)):
            print('bot', bot, 'top', top, 'y_intersect', y_1, 'left', left, 'right', right, 'x_intersect', x_1)
            print('coordinates out of target 1')
            return 1

        # repeat for second vertical
        if (linear_line[0] == vertical_one[0] or linear_line[0] == vertical_two[0]):
            print('parallel')
            return 1

        # get coordinates of intersection 
        x_2 = (vertical_two[1] - linear_line[1]) / (linear_line[0] - vertical_two[0])
        y_2 = vertical_two[0] * x_2 + vertical_two[1]
        left = (vertical_two[3] - vertical_two[1]) / vertical_two[0]
        right = (vertical_two[2] - vertical_two[1]) / vertical_two[0]
        top = vertical_two[2]
        bot = vertical_two[3]

        # check coordinates are within target
        if ((bot > y_2 or top < y_2) or (left > x_2 or right < x_2)):
            print('bot', bot, 'top', top, 'y_intersect', y_2, 'left', left, 'right', right, 'x_intersect', x_2)
            print('coordinates out of target 2')
            return 1
        
        return self.bow_height_intersection(((x_1, y_1), (x_2, y_2)), vertical_lines)

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
        bot_scaling_factor = .20
        top_scaling_factor = .10

        vertical_one = vertical_lines[0]
        vertical_two = vertical_lines[1]

        # get lower and upper limit from vertical points
        #bot_x1 = (vertical_one[3] - vertical_one[1]) / vertical_one[0] # x = (y - b) / m
        bot_y1 = vertical_one[3]
        #bot_x2 = (vertical_two[3] - vertical_two[1]) / vertical_two[0]
        bot_y2 = vertical_two[3]
        
        #top_x1 = (vertical_one[2] - vertical_one[1]) / vertical_one[0] # x = (y - b) / m
        top_y1 = vertical_one[2]
        #top_x2 = (vertical_two[2] - vertical_two[1]) / vertical_two[0]
        top_y2 = vertical_two[2]

        # get avg height of vertical lines
        height = (top_y1 - bot_y1 + top_y2 - bot_y2 ) / 2

        # get lower limit by averaging bottom y value. Scaled by height of strings and bot_scaling_factor
        min = ((bot_y1 + bot_y2) / 2) + height * bot_scaling_factor

        if (intersection_points[0][1] <= min or intersection_points[1][1] <= min):
            return 2
        
        # get upper limit by averaging top y value. Scaled by height of strings and bot_scaling_factor
        max = ((top_y1 + top_y2) / 2) - height * top_scaling_factor

        if (intersection_points[0][1] >= max or intersection_points[1][1] >= max):
            return 3
        
        return 0

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
            if point and len(point) == 4:
                points = [tuple(map(int,p)) for p in point]
                cv2.polylines(opencv_frame, [np.array(points)], isClosed=True, color=color, thickness=2)

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
    model = YOLO('new_obb2.pt')  # Replace with your actual model file    
    cap = cv2.VideoCapture("Student 1/supination and bow high - slow.mp4")

    def resize_keep_aspect(image, target_width=1200):
        """Resize image while keeping aspect ratio"""
        h, w = image.shape[:2]
        scale = target_width / w
        new_dim = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)

    cln = Classification()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv11 OBB inference
        results = model(frame)

        for result in results:
            if hasattr(result, 'obb') and result.obb is not None and result.obb.xyxyxyxy is not None:
                obb_coords = result.obb.xyxyxyxy.cpu().numpy()  # shape: (N, 4, 2)
                for box in obb_coords:
                    pts = box.reshape((-1, 1, 2)).astype(int)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            if len(result.obb.xyxyxyxy) >= 2:
                print("Both bow and string detected")
                if len(result.obb.xyxyxyxy) == 2:
                    if result.names[0] == result.names[1]:
                        continue #if both are bow or both are string, do nothing
                    if result.names[0] == "Bow": #first is bow, second is string
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
                        if result.names[0] == "Bow":
                            if result.obb[x].conf > bow_conf:
                                bow_conf = result.obb[x].conf
                                bow_index = x
                        elif result.names[0] == "String":
                            if result.obb[x].conf > string_conf:
                                string_conf = result.obb[x].conf
                                string_index = x
                    if bow_index != -1 and string_index != -1:
                        bow = torch.round(result.obb[bow_index])
                        string = torch.round(result.obb[string_index])
                bow_coords = [Point2D(bow[0][0].item(), bow[0][1].item()), Point2D(bow[1][0].item(), bow[1][1].item()), Point2D(bow[2][0].item(), bow[2][1].item()), Point2D(bow[3][0].item(), bow[3][1].item())]
                string_coords = [Point2D(string[0][0].item(), string[0][1].item()), Point2D(string[1][0].item(), string[1][1].item()), Point2D(string[2][0].item(), string[2][1].item()), Point2D(string[3][0].item(), string[3][1].item())]
                #cln.update_points(string_coords, bow_coords)
                #cln.get_vertical_lines()
                #annotated_frame = cln.display_classification(cln.intersects_vertical(), frame)
                

        


        # Resize frame for display
        resized_frame = resize_keep_aspect(frame, target_width=700)

        # Show the resized frame
        cv2.imshow("YOLOv11 OBB", resized_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

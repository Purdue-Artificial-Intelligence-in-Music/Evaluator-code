import new_classification_fixed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import random
import numpy as np

fig, ax = plt.subplots(figsize=(8, 6))

def rotated_rectangle(x, y, width, height, angle_degrees, color, alpha=1):
    """Create a rectangle rotated about its center."""
    rect = Rectangle((0, 0), width, height, color=color, alpha=alpha)
    
    # Create a transform that:
    # 1. Moves to the rectangle center
    # 2. Rotates
    # 3. Moves back to the original position
    center_x = x + width/2
    center_y = y + height/2
    transform = (Affine2D()
                 .translate(-width/2, -height/2)  # Move to origin for rotation
                 .rotate_deg(angle_degrees)        # Rotate
                 .translate(center_x, center_y)    # Move back to position
                 + ax.transData)
    
    rect.set_transform(transform)
    ax.add_patch(rect)
    return rect

def get_rotated_corners(x, y, width, height, angle_degrees):
    """Calculate the coordinates of the four corners after rotation"""
    # Original corners (before rotation)
    corners = np.array([
        [x, y],                     # Bottom-left
        [x + width, y],             # Bottom-right
        [x + width, y + height],    # Top-right
        [x, y + height]             # Top-left
    ])
    
    # Create rotation transform about the center
    center_x = x + width/2
    center_y = y + height/2
    rotation = (Affine2D()
                .translate(-center_x, -center_y)  # Move to origin
                .rotate_deg(angle_degrees)       # Rotate
                .translate(center_x, center_y))  # Move back
    
    # Apply the rotation to all corners
    rotated_corners = rotation.transform(corners)

    return rotated_corners

"""
1. Receive new bounding box corner coordinates.
2. Call update_points() with the new coordinates.
3. Compute and store vertical edge data using get_vertical_lines().
4. Call intersects_vertical() with the current line and vertical edges.
5. Interpret the return value from intersects_vertical():
  - 0 → Classify as "Correct"
  - 1 → Classify as "Outside Bow Zone"
  - 2 → Classify as "Too Low"
  - 3 → Classify as "Too High"
"""

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

def main():
    

    # bow
    bow_x = 30 * random.uniform(.95, 1.05)
    bow_y = 100 * random.uniform(.93, 1.07)
    bow_h = 3 * random.uniform(.8, 1.2)
    bow_w = 50 * random.uniform(.8, 1.2)
    bow_angle = random.randint(-40, 40)
    bow = rotated_rectangle(bow_x, bow_y, bow_w, bow_h, bow_angle, color='blue', alpha=0.5)
    bow_box = get_rotated_corners(bow_x, bow_y, bow_w, bow_h, bow_angle)

    # strings
    strings_x = 50 * random.uniform(.95, 1.05)
    strings_y = 90 * random.uniform(.95, 1.05)
    strings_h = 18 * random.uniform(.9, 1.1)
    strings_w = 10 * random.uniform(.9, 1.1)
    strings_angle = random.randint(-5, 5)
    strings = rotated_rectangle(strings_x, strings_y, strings_w, strings_h, strings_angle, color='red', alpha=0.5)
    strings_box = get_rotated_corners(strings_x, strings_y, strings_w, strings_h, strings_angle)

    # Set up the plot
    ax.set_xlim(20, 100)
    ax.set_ylim(50, 130)
    ax.set_aspect('equal')
    ax.set_title('Two Angled Rectangles')
    ax.grid(True)

    # classify the plot
    classifier = new_classification_fixed.Classification()
    # [top-left, top-right, bottom-right, bottom-left]
    string_box_xyxyxyxy = (strings_box[3], strings_box[2], strings_box[1], strings_box[0])
    # [botLeft, topLeft, topRight, botRight]
    bow_box_xyxyxyxy = (bow_box[3], bow_box[0], bow_box[2], bow_box[1])
    classifier.update_points(string_box_xyxyxyxy, bow_box_xyxyxyxy)
    midline = classifier.get_midline()
    vertical_lines = classifier.get_vertical_lines()
    height_result = classifier.intersects_vertical(midline, vertical_lines)
    angle_result = classifier.bow_angle(midline, vertical_lines)

    print('height_result:', height_label_map[height_result][0], '\nangle_result:', angle_label_map[angle_result][0])
    print('actual_bow_rotation:', bow_angle, 'actual_string_rotation:', strings_angle)
    plt.show()


if __name__ == '__main__':
    main()


import math
import numpy as np

#Calulates the angle between 4 points A,B,C,D
#Takes points A,B,C,D and returns the angle in degrees between two line segments AB and CD
def calculate_angle_between_lines(A, B, C, D):
    # Convert points to numpy arrays for easier vector operations
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)
    
    # Calculate vectors from the points
    vector1 = B - A  # Vector in direction of line AB
    vector2 = D - C  # Vector in direction of line CD
    
    # Calculate dot product
    dot_product = np.dot(vector1, vector2)
    
    # Calculate magnitudes
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate cosine of the angle
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    # Handle floating point errors
    cos_angle = min(1.0, max(-1.0, cos_angle))
    
    # Calculate angle in degrees
    angle_degrees = math.degrees(math.acos(cos_angle))
    
    return angle_degrees

def print_line_info(test_name, A, B, C, D):
    """Helper function to print line segment information and angle"""
    angle = calculate_angle_between_lines(A, B, C, D)
    print(f"\n{test_name}:")
    print(f"Line 1: A{A} to B{B}")
    print(f"Line 2: C{C} to D{D}")
    print(f"Angle: {angle:.2f} degrees")


# Test cases
if __name__ == "__main__":
    # Test Case 1: Perpendicular lines
    print_line_info("Perpendicular lines",
                   (0, 0), (1, 0),  # Horizontal line
                   (0, 0), (0, 1))  # Vertical line

    # Test Case 2: Parallel lines
    print_line_info("Parallel lines",
                   (0, 0), (1, 0),    # Horizontal line
                   (0, 2), (1, 2))    # Parallel horizontal line

    # Test Case 3: 45 degree angle
    print_line_info("45 degree angle",
                   (0, 0), (1, 0),    # Horizontal line
                   (0, 0), (1, 1))    # 45 degree line

    # Test Case 4: 60 degree angle
    print_line_info("60 degree angle",
                   (0, 0), (1, 0),            # Horizontal line
                   (0, 0), (0.5, 0.866))      # 60 degree line

    # Test Case 5: Random lines
    print_line_info("Random lines 1",
                   (2, 3), (5, 8),    # Random line 1
                   (1, 1), (4, 2))    # Random line 2

    # Test Case 6: Another random example
    print_line_info("Random lines 2",
                   (-1, -1), (2, 3),   # Random line 3
                   (0, 5), (4, 2))     # Random line 4

    # Test Case 7: Acute angle
    print_line_info("Acute angle",
                   (0, 0), (3, 1),     # Shallow angle line 1
                   (0, 0), (3, 2))     # Shallow angle line 2

    # Test Case 8: Obtuse angle
    print_line_info("Obtuse angle",
                   (0, 0), (1, 1),     # 45 degree line
                   (0, 0), (-1, 1))    # 135 degree line



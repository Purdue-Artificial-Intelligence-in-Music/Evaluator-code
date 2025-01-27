# AIM Evaluator: Installation and Troubleshooting Guide

## Overview
The Evaluator app, developed by the AIM ([Artificial Intelligence for Musicians](https://ai4musicians.org/#)) group at Purdue University, uses computer vision to assist string instrument players in improving their posture and hand movements. Refer to "Evaluator_Vision_team_Installation_and_Code_Setup_Documentation.pdf" for more details.

## Prerequisites

### 1. Git Installation
- Install Git from [Git's official website](https://git-scm.com/downloads).
- Verify installation with:
  ```
  git --version
  ```

### 2. Python Installation
- Ensure Python 3.9, 3.10, or 3.11 is installed. Download it from the [official Python website](https://www.python.org/downloads).
- Verify installation with:
  ```
  python --version
  ```

## Cloning the Repository
1. Clone the repository:
   ```
   git clone https://github.com/Purdue-Artificial-Intelligence-in-Music/Evaluator-code.git
   ```
2. Navigate to the project directory:
   ```
   cd Evaluator-code/src/computer_vision/hand_pose_detection
   ```

## Virtual Environment Setup

### 1. Create a Virtual Environment
```bash
python -m venv env
```

### 2. Activate the Virtual Environment
- **Windows**:
  ```
  .\env\Scripts\activate
  ```
- **macOS/Linux**:
  ```
  source env/bin/activate
  ```

## Installing Dependencies

### Install Required Packages
1. Ensure `pip` is updated:
   ```
   pip install --upgrade pip
   ```
2. Install the necessary dependencies:
   ```
   pip install opencv-python mediapipe numpy supervision ultralytics ipython
   ```

## Running the Script
1. Activate the virtual environment (if not already activated).
2. Run the script:
   ```
   python test.py
   ```

## Notes on File Paths
The following paths in your script should be updated for portability and to avoid hardcoding absolute paths:

### Current Paths
```python
gesture_model = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/3_category.task'
model = YOLO('/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/bow_target.pt')
video_file_path = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Vertigo for Solo Cello - Cicely Parnas.mp4'
```

### Updated Paths
```python
gesture_model = '3_category.task'
model = YOLO('bow_target.pt')
video_file_path = 'Vertigo for Solo Cello - Cicely Parnas.mp4'
```
Ensure that these files are placed in the same directory as your script or adjust the relative paths accordingly.

## Common Errors and Troubleshooting

### 1. Missing Module Errors
- **`ModuleNotFoundError: No module named 'cv2'`**:
  ```
  pip install opencv-python
  ```
- **`ModuleNotFoundError: No module named 'mediapipe'`**:
  ```
  pip install mediapipe
  ```
- **`ModuleNotFoundError: No module named 'supervision'`**:
  ```
  pip install supervision
  ```
- **`ModuleNotFoundError: No module named 'ultralytics'`**:
  ```
  pip install ultralytics
  ```
- **`ModuleNotFoundError: No module named 'IPython'`**:
  ```
  pip install ipython
  ```

### 2. File Not Found Errors
- **Model File (`best.pt`) Not Found**:
  Ensure the path to `best.pt` is correct. Use relative paths to the project root or ensure the file exists in the specified location.
- **Video File Not Found**:
  Verify the file exists at the specified path. Update the script to use the correct relative or absolute path.

## Additional Notes
- Always activate your virtual environment before running the script.
- Ensure all dependencies are installed and up to date.
- Double-check file paths for models and input files to avoid runtime errors.

For further assistance, consult the project documentation or reach out to the project team leads.


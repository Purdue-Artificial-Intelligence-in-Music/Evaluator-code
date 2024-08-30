# env setup commands:
    # create conda env and activate
    # conda install -c conda-forge python=3.10 pip numpy opencv
    # pip install -r requirements.txt


# import libraries
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import os
import supervision as sv
import ultralytics
from ultralytics import YOLO
from IPython.display import display, Image

# DEFINE CONSTANTS
# Must download
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
hand_task_file = os.path.dirname(os.path.realpath(__file__)) + '/../../../models/hand_landmarker.task'
# https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
body_task_file = os.path.dirname(os.path.realpath(__file__))  + '/../../../models/pose_landmarker.task'

verbose = 0

# Base option setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Hand Landmarker
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
# Pose Landmarker
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
# Global frame count
current_frame = 0

# A static dictionary which allows for the lookup of the
# indices corresponding to each hand
lines: dict[str, list[int]] = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20]
}


def telescoping(values: list[int]) -> list[(int, int)]:
    """
    Creates a `telescoping` version of the values passed in pairing each
    value with its successor.
    For example:
    >>> telescoping([1,2,3])
    [(1, 2), (2, 3)]

    :param values: the list of values to convert to a telescoping list
    :return: the telescoped version of `values`
    """
    result = []
    for i in range(0, len(values) - 1):
        result.append((values[i], values[i + 1]))
    return result


# Pre-computed connections and nodes derived from `lines`
hand_connections = [pair for x in lines for pair in telescoping(lines[x])]
hand_nodes = [node for finger in lines for node in lines[finger]]


def distance(a: (int, int), b: (int, int)) -> float:
    """
    Calculates the euclidian distance between two points
    :param a: the first point
    :param b: the second point
    :return: the distance between a and b using Pythagoras' theorem
    """
    ax, ay = a
    bx, by = b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def straightness(points: [int], landmarks: list[(int, int)]) -> float:
    """
    Calculates the `straightness` of a segment of points
    This straightness is essentially the same as sinuosity
    (https://en.wikipedia.org/wiki/Sinuosity)

    :param points: the indices of the landmarks to consider in order of the intended line to be formed
    :param landmarks: the full list of points, some of which are considered
    :return: the straightness of the line, in (0, 1], where the higher values indicate a straighter line formed
    """
    # select the relevant segments
    lm = [landmarks[i] for i in points]
    d_running = 0
    for i in range(0, len(points) - 1):
        d_running += distance(lm[i], lm[i + 1])
    # total
    dtot = distance(lm[0], lm[-1])
    return dtot / d_running


# Data structure for persistent storae of the result
# TODO: probably convert this to a instance variable of a class
current_hand_result: list[HandLandmarkerResult] = []
current_pose_result: list[PoseLandmarkerResult] = []


def handle_result_hand(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
    Handles the result of model by saving it into the global `current_result`
    :param result: the result obtained

    :param timestamp_ms: IGNORED (needed for calling convention)
    :param output_image: IGNORED (needed for calling convention)
    """
    current_hand_result.clear()
    if len(result.handedness) != 0:
        current_hand_result.append(result)


def handle_result_pose(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
        Handles the result of model by saving it into the global `current_result`
        :param result: the result obtained

        :param timestamp_ms: IGNORED (needed for calling convention)
        :param output_image: IGNORED (needed for calling convention)
        """
    current_pose_result.clear()
    if len(result.pose_landmarks) > 0:
        current_pose_result.append(result)


# Setup options for the model

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=hand_task_file),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result_hand)
pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=body_task_file),
    output_segmentation_masks=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result_pose
)


def get_position(lm: NormalizedLandmark, width: int, height: int) -> tuple[int, int]:
    """
    Maps a landmark to a pixel
    :param lm the landmark to use (with x,y in [0,1])
    :param width: the image width
    :param height: the image height
    :return: the pixel position of the landmark
    """
    x = int(width * lm.x)
    y = int(height * lm.y)
    return x, y


def main():
    """
    Main function that sets up video feed, runs model, and displays livestream
    """
    model = YOLO('/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Vertigo for Solo Cello - Cicely Parnas.mp4')  # Path to your model file
    
    with PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
        with HandLandmarker.create_from_options(hand_options) as hand_landmarker:
            video_file_path = '/Users/Wpj11/Documents/GitHub/Evaluator-code/src/computer_vision/hand_pose_detection/Too much pronation (1).mp4'
            # get video capture
            video_capture = cv2.VideoCapture(video_file_path)
            frame_count = 0

            # save as video
            writer = cv2.VideoWriter("demo.avi", cv2.VideoWriter_fourcc(*"MJPG"), 12.5,(640,480)) # algo makes a frame every ~80ms = 12.5 fps

            while True:
                # get next frame
                ret, frame = video_capture.read()
                frame_count += 1

                # frame = cv2.flip(frame, 1)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                hand_landmarker.detect_async(mp_image, frame_count)
                pose_landmarker.detect_async(mp_image, frame_count)

                # yolov8 prediction
                results = model(frame)
                detections = sv.Detections.from_ultralytics(results[0])

                # display the frame
                frame = np.copy(frame)
                height = len(frame)
                width = len(frame[0])
                frame = cv2.putText(
                    frame,
                    "Frame {}".format(frame_count),
                    (10, 50),
                    cv2.QT_FONT_NORMAL,
                    1,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                dy = 25
                yPos = 50
                if len(current_pose_result) != 0:
                    result = current_pose_result[0]
                    for cur in result.pose_landmarks:
                        landmarks = [get_position(x, width, height) for x in cur]
                        for lm in landmarks[11:15] + landmarks[23:29]: # excludes the face and hand pose landmarks
                            frame = cv2.circle(frame, lm, radius=10, color=(255, 0, 0), thickness=-1)

                if len(current_hand_result) != 0:
                    result = current_hand_result[0]
                    for hands in result.handedness:
                        yPos += dy
                        category = hands[0]
                        name = category.category_name
                        score = int(category.score * 100)
                        output = "Found {} hand ({}%)".format(name, score)
                        frame = cv2.putText(
                            frame,
                            output,
                            (10, yPos),
                            cv2.QT_FONT_NORMAL,
                            dy / 50,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                    for hand_pose_data in result.hand_landmarks:
                        landmarks = [get_position(x, width, height) for x in hand_pose_data]
                        # draw dots
                        for i in hand_nodes:
                            frame = cv2.circle(frame, landmarks[i], radius=5, color=(0, 255, 255), thickness=-1)
                        # draw connectors
                        for (iA, iB) in hand_connections:
                            a = landmarks[iA]
                            b = landmarks[iB]
                            frame = cv2.line(frame, a, b, color=(0, 255, 255), thickness=1)
                        # get straightness params (basic analytics)
                        # if (verbose):
                        #     for name in lines:
                        #         points = lines[name]
                        #         yPos += dy
                        #         s = straightness(points, landmarks)
                        #         output = "Finger {} straightness: {:.2f}".format(name, s)
                        #         frame = cv2.putText(
                        #             frame,
                        #             output,
                        #             (10, yPos),
                        #             cv2.QT_FONT_NORMAL,
                        #             dy / 50,
                        #             (255, 0, 0),
                        #             1,
                        #             cv2.LINE_AA,
                        #         )
                        yPos += dy

                # add bounding boxes
                oriented_box_annotator = sv.OrientedBoxAnnotator()
                annotated_frame = oriented_box_annotator.annotate(
                    scene=frame,
                    detections=detections
)

                # sv.plot_image(image=frame, size=(16, 16))
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # cv2.resizeWindow('frame', 2560,1440)
                writer.write(frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # clean up
            video_capture.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

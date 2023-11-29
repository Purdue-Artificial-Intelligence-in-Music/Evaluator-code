# import libraries
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# Video reading code adapted from:
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/

# Base option setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

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
connections = [pair for x in lines for pair in telescoping(lines[x])]
nodes = [node for finger in lines for node in lines[finger]]


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
current_result: list[HandLandmarkerResult] = []


def handle_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
    """
    Handles the result of model by saving it into the global `current_result`
    :param result: the result obtained

    :param timestamp_ms: IGNORED (needed for calling convention)
    :param output_image: IGNORED (needed for calling convention)
    """
    current_result.clear()
    if len(result.handedness) != 0:
        current_result.append(result)


# Setup options for the model
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../../../models/hand_landmarker.task'),
    num_hands=2,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result)


def get_position(lm: NormalizedLandmark, width: int, height: int) -> (int, int):
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
    with HandLandmarker.create_from_options(options) as landmarker:
        # get video capture
        video_capture = cv2.VideoCapture(0)
        frame_count = 0
        while True:
            # get next frame
            ret, frame = video_capture.read()
            frame_count += 1
            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            landmarker.detect_async(mp_image, frame_count)
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
            dy = 50
            yPos = 50
            if len(current_result) != 0:
                result = current_result[0]
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
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

                for hand_pose_data in result.hand_landmarks:
                    landmarks = [get_position(x, width, height) for x in hand_pose_data]
                    # draw dots
                    for i in nodes:
                        frame = cv2.circle(frame, landmarks[i], radius=5, color=(0, 255, 255), thickness=-1)
                    # draw connectors
                    for (iA, iB) in connections:
                        a = landmarks[iA]
                        b = landmarks[iB]
                        frame = cv2.line(frame, a, b, color=(0, 255, 255), thickness=1)
                    # get straightness params (basic analytics)
                    for name in lines:
                        points = lines[name]
                        yPos += dy
                        s = straightness(points, landmarks)
                        output = "Finger {} straightness: {:.2f}".format(name, s)
                        frame = cv2.putText(
                            frame,
                            output,
                            (10, yPos),
                            cv2.QT_FONT_NORMAL,
                            1,
                            (255, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # clean up
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

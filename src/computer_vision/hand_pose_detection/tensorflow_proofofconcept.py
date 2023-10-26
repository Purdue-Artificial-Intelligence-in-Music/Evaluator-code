# import libraries
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# Video reading code adapted from:
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_frame = 0

connections = [
    # along thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # base of palm to base of each finger
    (0, 5),
    (0, 9),
    (0, 13),
    (0, 17),
    # index finger
    (5, 6), (6, 7), (7, 8),
    # middle finger
    (9, 10), (10, 11), (11, 12),
    # ring finger
    (13, 14), (14, 15), (15, 16),
    # pinky finger
    (17, 18), (18, 19), (19, 20)
]

lines: dict[str, list[int]] = {
    "thumb": [0, 1, 2, 3],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}


def distance(a: (int, int), b: (int, int)) -> float:
    ax, ay = a
    bx, by = b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def straightness(points: [int], landmarks: list[(int, int)]) -> float:
    # segment distances
    lm = [landmarks[i] for i in points]
    d_running = 0
    for i in range(0, len(points) - 1):
        d_running += distance(lm[i], lm[i+1])
    # total
    dtot = distance(lm[0], lm[-1])
    return dtot / d_running



current_result: list[HandLandmarkerResult] = []


# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    current_result.clear()
    if len(result.handedness) != 0:
        current_result.append(result)


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='../../../models/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)


def get_position(lm: NormalizedLandmark, width: int, height: int) -> (int, int):
    x = int(width * lm.x)
    y = int(height * lm.y)
    return x, y


def main():
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
            frame = cv2.putText(frame, "Frame {}".format(frame_count), (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 1,
                                cv2.LINE_AA)
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
                i = 0
                landmarks = [get_position(x, width, height) for x in result.hand_landmarks[0]]
                # draw dots
                for i, pos in enumerate(landmarks):
                    frame = cv2.circle(frame, pos, radius=5, color=(0, 255, 255), thickness=-1)
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

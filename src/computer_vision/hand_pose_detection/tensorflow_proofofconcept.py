# import libraries
import cv2
import mediapipe as mp
import numpy as np

# Video reading code adapted from:
# https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/


BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

current_frame = 0

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
            frame = cv2.putText(frame, "Frame {}".format(frame_count), (10, 50), cv2.QT_FONT_NORMAL, 1, (0, 0, 255), 1, cv2.LINE_AA)
            if len(current_result) != 0:
                dy = 50
                y = 50
                result = current_result[0]
                for hands in result.handedness:
                    y += dy
                    category = hands[0]
                    name = category.category_name
                    score = int(category.score * 100)
                    output = "Found {} hand ({}%)".format(name, score)
                    frame = cv2.putText(
                        frame,
                        output,
                        (10, y),
                        cv2.QT_FONT_NORMAL,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                for landmark in result.hand_landmarks[0]:
                    x = int(width * landmark.x)
                    y = int(height * landmark.y)
                    frame = cv2.circle(frame, (x, y), radius=5, color=(0, 255, 255), thickness=-1)

            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # clean up
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

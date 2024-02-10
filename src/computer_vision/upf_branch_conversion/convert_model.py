import cv2

##This script detects faces in each frame of the video
##using the Haar cascade classifier, extracts the faces, and saves them as individual image files.

##can be altered to detect instruments and limbs (object detection --> YOLO)

## Replace "input_video.mp4" with the path to video file 
## and "haarcascade_frontalface_default.xml" with the path 
## to the XML file containing the pre-trained face detection model  --> 
def main():
    cap = cv2.VideoCapture('input_video.mp4')
    if not cap.isOpened():
        print("Error opening video file")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            cv2.imshow('Face', face_roi)
            cv2.imwrite(f'face_{frame_count}.jpg', face_roi)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
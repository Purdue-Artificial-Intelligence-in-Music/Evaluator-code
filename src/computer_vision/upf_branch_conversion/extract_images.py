import cv2

# This (supporting) script detects faces in each frame of a video using the Haar cascade classifier, 
# extracts the faces, and saves them as individual image files.

# Can be altered to detect instruments and limbs using a trained object detection model

# Replace "input_video.mp4" with the path to video file 
# and "haarcascade_frontalface_default.xml" with the path to the XML file containing the pre-trained object detection model

def main():
    cap = cv2.VideoCapture('input_video.mp4')
    if not cap.isOpened():
        ##unable to open
        print("Error opening video file")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    
    # Extracts frames containing faces for every frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # For each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (face) from the frame
            face_roi = frame[y:y+h, x:x+w]
            
            # Display the extracted face
            cv2.imshow('Face', face_roi)
            # Save the extracted face as an image file
            cv2.imwrite(f'face_{frame_count}.jpg', face_roi)

        frame_count += 1

        # Check for user input to exit loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

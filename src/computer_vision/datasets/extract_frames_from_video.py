import cv2
import os

#os for file handling and cv2 for opencv, which enables us to extract frames from video

def extract_frames(video_path, output_folder):
    #check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #else, capture video using opencv and keep track of number of frames
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        #capture frames in frame
        captured, currFrame = cap.read()

        if not captured:
            #captured is set to true when frame is read correctly
            print(f"frame {frame_count} has not been read correctly")
            break

        #save frame as an image file in output folder
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, currFrame)

        frame_count += 1

    #release the video capture
    cap.release()
    print(f"Successfully extracted {frame_count} frames and saved to '{output_folder}'!")

#usage
video_path = 'April_video_demo.mp4'
output_folder = 'Photos'
extract_frames(video_path, output_folder)

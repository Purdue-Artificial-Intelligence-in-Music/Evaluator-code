import cv2
import os

#Calculates the frame rate of the video and extracts one frame every 3 seconds (can be varied)

def extract_frames(video_path, output_folder, interval=3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)

    #Get frames per second
    fps = int(cap.get(cv2.CAP_PROP_FPS)) 

    #Calculate the interval in frames
    frame_interval = fps * interval  
    
    frame_count = 0
    saved_frames = 0
    
    while True:
        captured, currFrame = cap.read()
        
        if not captured:
            print(f"Frame {frame_count} has not been read correctly or end of video reached.")
            break
        
        #Save image for our use
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_path, currFrame)
            saved_frames += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Successfully extracted {saved_frames} frames and saved to '{output_folder}'!")

# Usage
video_path = 'C:\\Users\\Gurte\\Downloads\\right posture 2.mp4'
output_folder = 'Photos'
extract_frames(video_path, output_folder)

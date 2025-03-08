from pytube import YouTube
import cv2
import os

#os for file handling and cv2 for opencv, which enables us to extract frames from video

'''
Had to get some permissions for the YouTube video for this to work.
YouTube API policy is weird and changes and this may not work for everything.
If this doesn't work, the best approach may be to just download the video(third party app/extension/website) 
and use extracting_frames_from_video.py, which should work 100% no problem.
'''
def download_youtube_video(url, download_folder):
    #download url and save in folder
    video = YouTube(url)
    video_stream = video.streams.filter(file_extension = 'mp4', progressive=True).first()
    video_path = video_stream.download(output_path = download_folder)
    return video_path

def extract_frames(video_path, output_folder):
    #check if folder exists
    if  not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #Now extract frames from video at path and save in folder
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # save each frame in frame
        ret, frame = cap.read()

        if not ret:
            #if frame is read correctly, ret is set to true
            break

        #save frame as an image file
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_path, frame)

        #update frame count for labels in folder
        frame_count += 1

    # release the video capture
    cap.release()
    print(f"Extracted {frame_count} frames and saved all to '{output_folder}'.")


#main function that calls all of these
def extract_frames_from_youtube_vid(url, output_folder):
    #download youtube video
    download_folder = "youtube_downloads"

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    video_path = download_youtube_video(url, download_folder)

    #extract frames from video
    extract_frames(video_path, output_folder)

    #delete downloaded video to save space
    os.remove(video_path)

#actually using this
AIM_vid = "https://www.youtube.com/watch?v=Qfqyhm134pU"
output_folder = "Photos"
extract_frames_from_youtube_vid(AIM_vid, output_folder)


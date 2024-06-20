# Python Script for extracting video frames from Solos Video Data and convert
# the frames and store them in a folder as JPGs. The number of frames and 
# dimension of the frames can be specified by the user.

#pip install pytube
import cv2
import os
# from pytube import YouTube
import random



def extract_frames(youtube_video_id, output_folder, frame_per_second,
                   image_crop_check, dimension_x = None, dimension_y = None,
                   start_time = 20, fps=1):
    """ Downloads video given an id that will be used to extract frames from.
    The video will delete once it's done being extracted. Any problems with
    downloading YT video is accounted for.

    Args:
      youtube_video_id: Defines a Youtube ID for grabbing a specific
      Youtube Videos (YT vids).
      output_folder: the path of a folder that will be used to download
      YT vids and stores frames in.
      frames_per_second: Boolean if user wants to extract 1 frame for every
      second or not.
      image_crop_check: Boolean if user would like frames to be cropped to a
      specific dimension.
      dimension_x: Given that user wants to crop frames to a certain size,
      this variable defines the width the frame should be cropped to.
      dimension_y: Given that user wants to crop frames to a certain size, 
      this variable defines the height the frame should be cropped to.
    
      Returns:
        Nothing.
    """
    try:
      # Download the YouTube video if possible
      yt = YouTube(f"https://www.youtube.com/watch?v={youtube_video_id}")
      video = yt.streams.get_highest_resolution()
      if not video:
          print("Video not found.")
          return

      # Stores downloaded video in specific folder
      video_path = video.download(output_path=output_folder)
      video_filename = os.path.basename(video_path)

      # Use ffmpeg to trims the beginning of the video until the start_time
      output_filename = os.path.join(output_folder, f"trimmed_{video_filename}")
      os.system(f'ffmpeg -ss {start_time} -i "{video_path}" ' +
                                f'-c copy "{output_filename}"')

      # Extract frames from the downloaded video
      extract_frames_from_video(output_filename, output_folder, 
                                frame_per_second, image_crop_check, 
                                dimension_x, dimension_y)

      # Delete the downloaded video file
      os.remove(output_filename)

    except Exception as e:
      print(f"An error occurred: {str(e)}")

# Grabs random unique integers within 0 to max number of frames per a video
def generate_unique_random_list(start, end, number_of_frames):
    """Generates a list of unique integers within a specific range and 
    specific number of elements.

    Args:
        start: the minimum value an integer must be equal to or greater than.
        end: the maximum value an integer must be equal to or less than.
        number_of_frames: the number of unique integers the user would like to
        be generated in the list.

    Returns:
        inique_list: A randomy list of unique (non-repetitive) integers within
        avgiven range.

    Raises:
        ValueError: Length of list cannot exceed the range of integers.
    """
    if number_of_frames > (end - start + 1):
        raise ValueError("Length of list cannot exceed the range of integers.")
    
    # Generates random list by first defining range and then number of frames
    unique_list = random.sample(range(start, end + 1), number_of_frames)
    return unique_list

def crop_to_center(image):
    """Crops the outer 20% of the images leaving only 60% of the middle of 
    the image.

    Args:
        image: A frame extracted from a YT vid that the user wants to crop

    Returns:
        cropped_square: a square croped image of the middle 60% of the 
        original image.
    """
    # Stores values of image dimension in height and width variable
    height, width, _ = image.shape
    # Define the portion of the image you want to keep (60% of the middle)
    keep_percentage = 0.6
    exclude_percentage = (1 - keep_percentage) / 2  # 20% from each side
    start_x = int(width * exclude_percentage)
    end_x = int(width * (1 - exclude_percentage))

    # Keep all rows, crop columns from start_x to end_x
    cropped_image = image[:, start_x:end_x]
    # Determine the size of the square (same width and height)
    min_dimension = min(cropped_image.shape[0], cropped_image.shape[1])
    # Crop the square
    cropped_square = cropped_image[:min_dimension, :min_dimension]
    return cropped_square

def crop_to_square(image, dimension_x, dimension_y):
    """Crops an extracted frame image given specific dimensions from the user
    to crop image from the center.

    Args:
        image: A frame extracted from a YT vid that the user wants to crop
        dimension_x: the width the user would like the image to crop down to
        dimension_y: the height the  user would like the image to crop down to

    Returns:
        cropped_resized_image: the newly cropped image
    """
    # Extracting height and width from the image shape
    height, width, _ = image.shape

    # Determine the minimum dimension (height or width) of the image
    min_dimension = min(height, width)

    # Calculate the starting point for cropping the image to make it square
    start_x = (width - min_dimension) // 2
    start_y = (height - min_dimension) // 2

    # Crop the image to make it square
    cropped_image = image[start_y:start_y+min_dimension, 
                          start_x:start_x+min_dimension]


    # Resize the cropped image to the desired dimensions
    cropped_resized_image = cv2.resize(cropped_image, 
                                      (dimension_x, dimension_y))

    # Return the cropped and resized image
    return cropped_resized_image

def extract_frames_from_video(video_path, output_folder, frame_per_second, 
                              image_crop_check, dimension_x, dimension_y):

    """ Searches for YT vid in folders and begins extracting frames and 
    storing in folders. Based on users preference for arguments passed, 
    user can customize how frames should be extracted.


    Args:
      video_path: the path of a downloaded YT vid user would like to extract 
      frames from.
      output_folder: the path of a folder that will be used to download YT vids
      and stores frames in.
      frames_per_second: Boolean if user wants to extract 1 frame for every 
      second or not.
      image_crop_check: Boolean if user would like frames to be cropped to a 
      specific dimension.
      dimension_x: Given that user wants to crop frames to a certain size, 
      this variable defines the width the frame should be cropped to.
      dimension_y: Given that user wants to crop frames to a certain size, 
      this variable defines the height the frame should be cropped to.
    
      Returns:
        Nothing.
    """

    # Open the video file for reading
    vidcap = cv2.VideoCapture(video_path)

    # Read the first frame of git ftthe video
    success, image = vidcap.read()

    # Calculate total number of frames in the video and
    #generate a list of random integers representing frame indices
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_list = generate_unique_random_list(1, total_frames, 10)

    # If frame_per_second flag is set
    if frame_per_second:

      fps = vidcap.get(cv2.CAP_PROP_FPS)
      # Iterate through each randomly selected frame
      for frame_number in random_frame_list:
          # Calculate the time position in seconds for the current frame
          time_position_seconds = frame_number / fps

          # Set the position in the video to the calculated time position
          vidcap.set(cv2.CAP_PROP_POS_MSEC, time_position_seconds * 1000)

          # Read the frame at the calculated position
          success, image = vidcap.read()

          # If frame is successfully read
          if success:

              # Define the filename for the frame and write the frame to file
              frame_filename = os.path.join(output_folder, 
                                            f"frame_{frame_number}.jpg")
              cv2.imwrite(frame_filename, image)

              # Print a message indicating successful extraction of the frame
              print(f"Frame {frame_number} extracted")

          else:
              # Print an error message if frame extraction fails
              print(f"Error extracting frame {frame_number}")

    # If frame_per_second flag is not set
    else:
      # Calculate the number of frames to extract (30% of total frames)
      frames_to_extract = int(0.3 * total_frames)

      # Randomly sample frame indices without replacement
      frame_indices = random.sample(range(total_frames), frames_to_extract)

      # Iterate through each randomly selected frame index
      for i, frame_index in enumerate(frame_indices):


          vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

          # Read the frame at the current position
          success, image = vidcap.read()

          # If frame is not successfully read, break the loop
          if not success:
              break

          # If image_crop_check flag is set, crop the image to the center
          if image_crop_check:
              image = crop_to_center(image)

          # Define the file path for saving frames and write the frames to file
          image_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
          cv2.imwrite(image_path, image)

          # Print a message indicating the successful extraction of the frame
          print(f"Saved frame {i+1}/{frames_to_extract} as {image_path}")

    # Release the video capture object
    vidcap.release()

def main(video_path, output_path):
    extract_frames(video_path, output_path, True, True)

if __name__ == "__main__":
    video_path = "/video_dataset/Right elbow too high.mp4"
    output_path = "/posture_video_dataset"
    main(video_path, output_path)

# def main(output_path):
#     """Starts whole proces of extract frames from a list of videos 
#     with options of how frames should be extracted in terms of number 
#     of frames and dimension of frames.
    
#     Args:
#       output_path: folder path for video frames to be exported and 
#       stored into as JPGs.

#     Returns:
#       No return.
#     """

#     # List of unique Youtube IDs that will be used to search and download the 
#     # videos
#     yt_ids = [
#         "3TR7YwNESt0", "GkT1NLMY7xQ", "b28_gkUgqoE", "GMyL9JTfDps", 
#         "iNef2eekeK0", "QPfE-FqhQq4", "P--7W1h4khQ", "N7te_FRdSgk", 
#         "0jSLgzeFmao", "IpCafmoaImc", "PR1Pzaqew9w", "SPh_q2n3JYM", 
#         "ww9_f6hxYKo", "REhp68K-6mw", "l9ufvevHbkE", "TiqUpTAwWgY", 
#         "CNDZfj3BaqA", "voJaSCe8UyM", "UWRZ4gi3HAM", "5IwWfy2RhQE", 
#         "VEaJzYRrrD8", "nAtHjkKGr4A", "MKsYR-9ZD9M", "07kcsRHipyw", 
#         "Q0OfnLEpty4", "6--mQjcImjM", "3PrFqTav5Ow", "WSheIMbyC5w", 
#         "uIqbE0Ylh9o", "ebMlp9Kc_bY", "OvXAT3_GGec", "LDq9sFUjwQk", 
#         "J_nJYlZHpuw", "wXq_IU_vvrI", "Z1rRodHDapI", "6GEkwyK_tns", 
#         "s92f3CW9IdQ", "MHhK6jG6U2I", "vI-A3KJSBpk", "ZNs5igA6tZQ", 
#         "JxhUFn5CT6A", "iXnGftY7k_k", "6XJCvBBQu78", "BvrGLXMvc_o", 
#         "t81IJJVi-Us", "7nzwswUUDto", "-ZMb-QiqDJY", "zGiFcVKafQM", 
#         "hC7XPGO45hk", "VdFnyZWJAgo", "eRpbaoIOJjI", "dfPeeHyfpP8", 
#         "2C7_o5mfE6k", "dKBVWebVakI", "gRy76nNP6CQ", "mGQLXRTl3Z0", 
#         "PDJ_QZAbGi4", "w7Kh0LscP3U", "1u3yHICR_BU", "sBZJCnxdmPc", 
#         "yC8ovVGeUfA", "DpvxiC0osbA", "MNB8H2p9Kk0", "5sXcGZ60NpI", 
#         "kG3CrRcMgMk", "dTmuOKUPsYI", "RDnLgY2-abc", "WotHO4X1kak", 
#         "I0LedcEaPL0", "7WeXhVxpTAw", "JW9-YhFMYSE", "8PXQeH0g6jI", 
#         "wHfhjkI2L9M", "4UI2fPRaxpg", "7_8yPpCjK_w", "aLTH6DVXAcU", 
#         "KtsdeqrcO2U", "iuvaHDIGP5U", "mmJim2h-byQ", "FR1UZKD238Y", 
#         "42ULINLSfrE", "A2d2DHxV4l0", "4jxmrdkyjCc", "09iOwYycY6w", 
#         "AT97pTC97tY", "j9Re7hWGh3Y", "Zz9nwbi9oqA", "j4CwTQmRxiY", 
#         "vWytPM5Wawk", "WSvQ7_N1y08", "llpPao0uM7U", "-qRn8UyHogA", 
#         "MbuRvmZy3Yw", "MaYRimw-mx8", "Ng8QfKTk1KA", "seWlG50y-so", 
#         "fLH9GyDbUjU", "bw2Etrk6cHg", "fJ7yh1E9S-k", "1X2l_1za1g0", 
#         "DsQX--nVPJQ", "qxiDk7x6gkM", "BJW31sWO5cE", "CrBCIFZImKE", 
#         "3qHdCTe0mIs", "4-TAbbQmCtg", "qtIrc7YUEvo", "A2KrKTMSoEU", 
#         "Jsu5p6iAApI", "xFlBZMiVMT0", "qtVf-KkpX1w", "DEmhfGTWt8g", 
#         "PCicM6i59_I", "poCw2CCrfzA", "7Z96EdnfzoQ", "Yi9Xdiq579o", 
#         "qzl9w7gME50", "Vp2lSAv__wE", "VlIcqDWmPkw", "gtr9N_qdHag", 
#         "rifOR2Q0cRk", "bhxNofWXTEA", "aHiqoSUV5MA", "DB3kIKJPdUA", 
#         "RV4Ob2Y9hvQ", "BObT8M5Mnus", "bNLswbcgbF4", "stgIDZN6e5Q", 
#         "eWTR-puxJyo", "Y1ZO2qfr0ko", "D_a1o8biIAk", "PpcSlOZY3Aw", 
#         "heFoLuid8-Y", "JQOC7Za4Pbw"
#     ]
#     #Iterates through yt_ids where frames will be extracted from into jpgs
#     for i in range(0, len(yt_ids)):
#         print(f"Video {i}")

#         # Function that begins extracting frame process
#         extract_frames(yt_ids[i], "/content/SoloCelloFramesSetter", True, True)


# if __name__ == "__main__":
    
#     #Creates folder path for video frames to be exported and stored in as JPGs
#     output_path = "/content/SoloCelloFramesSetter"
#     main(output_path)

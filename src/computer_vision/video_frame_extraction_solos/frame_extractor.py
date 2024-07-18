import cv2
import os
import random

def extract_frames(video_path, output_folder, frame_per_second,
                   image_crop_check, dimension_x=None, dimension_y=None,
                   start_time=7, fps=1):
    """
    Extracts frames from a given video file and saves them as JPGs in a specified folder.
    The number of frames and dimensions of the frames can be specified by the user.

    Args:
      video_path: Path to the video file to extract frames from.
      output_folder: The path of a folder that will store frames.
      frame_per_second: Boolean if user wants to extract 1 frame for every second or not.
      image_crop_check: Boolean if user would like frames to be cropped to a specific dimension.
      dimension_x: Given that user wants to crop frames to a certain size, this variable defines the width the frame should be cropped to.
      dimension_y: Given that user wants to crop frames to a certain size, this variable defines the height the frame should be cropped to.
      start_time: Time in seconds from where to start extracting frames (default is 20).
      fps: Frames per second rate of the video (default is 1).
    """
    try:
        # Check if the video file exists
        if not os.path.exists(video_path):
            print("Video file not found.")
            return

        # Generate the output filename for the trimmed video
        video_filename = os.path.basename(video_path)
        output_filename = os.path.join(output_folder, f"trimmed_{video_filename}")

        # Use ffmpeg to trim the beginning of the video until the start_time
        os.system(f'ffmpeg -ss {start_time} -i "{video_path}" -c copy "{output_filename}"')

        # Extract frames from the trimmed video
        extract_frames_from_video(output_filename, output_folder,
                                  frame_per_second, image_crop_check,
                                  dimension_x, dimension_y)

        # Delete the trimmed video file
        os.remove(output_filename)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def generate_unique_random_list(start, end, number_of_frames):
    """
    Generates a list of unique integers within a specific range and specific number of elements.

    Args:
        start: The minimum value an integer must be equal to or greater than.
        end: The maximum value an integer must be equal to or less than.
        number_of_frames: The number of unique integers the user would like to be generated in the list.

    Returns:
        unique_list: A random list of unique (non-repetitive) integers within a given range.

    Raises:
        ValueError: Length of list cannot exceed the range of integers.
    """
    if number_of_frames > (end - start + 1):
        raise ValueError("Length of list cannot exceed the range of integers.")

    unique_list = random.sample(range(start, end + 1), number_of_frames)
    return unique_list


def crop_to_center(image):
    """
    Crops the outer 20% of the images leaving only 60% of the middle of the image.

    Args:
        image: A frame extracted from a video that the user wants to crop.

    Returns:
        cropped_square: A square cropped image of the middle 60% of the original image.
    """
    height, width, _ = image.shape
    keep_percentage = 0.6
    exclude_percentage = (1 - keep_percentage) / 2
    start_x = int(width * exclude_percentage)
    end_x = int(width * (1 - exclude_percentage))

    cropped_image = image[:, start_x:end_x]
    min_dimension = min(cropped_image.shape[0], cropped_image.shape[1])
    cropped_square = cropped_image[:min_dimension, :min_dimension]
    return cropped_square


def crop_to_square(image, dimension_x, dimension_y):
    """
    Crops an extracted frame image given specific dimensions from the user to crop image from the center.

    Args:
        image: A frame extracted from a video that the user wants to crop.
        dimension_x: The width the user would like the image to crop down to.
        dimension_y: The height the user would like the image to crop down to.

    Returns:
        cropped_resized_image: The newly cropped image.
    """
    height, width, _ = image.shape
    min_dimension = min(height, width)
    start_x = (width - min_dimension) // 2
    start_y = (height - min_dimension) // 2

    cropped_image = image[start_y:start_y + min_dimension, start_x:start_x + min_dimension]

    # Validate and handle dimension_x and dimension_y
    if dimension_x is None or dimension_x <= 0:
        dimension_x = min_dimension  # or some default value
    if dimension_y is None or dimension_y <= 0:
        dimension_y = min_dimension  # or some default value

    cropped_resized_image = cv2.resize(cropped_image, (dimension_x, dimension_y))
    return cropped_resized_image



def extract_frames_from_video(video_path, output_folder, frame_per_second, image_crop_check, dimension_x, dimension_y):
    """
    Extracts frames from the video file and saves them as JPGs in a specified folder.

    Args:
      video_path: The path of a downloaded video file user would like to extract frames from.
      output_folder: The path of a folder that will be used to store frames in.
      frame_per_second: Boolean if user wants to extract 1 frame for every second or not.
      image_crop_check: Boolean if user would like frames to be cropped to a specific dimension.
      dimension_x: Given that user wants to crop frames to a certain size, this variable defines the width the frame should be cropped to.
      dimension_y: Given that user wants to crop frames to a certain size, this variable defines the height the frame should be cropped to.
    """
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    random_frame_list = generate_unique_random_list(1, total_frames, 100)

    if frame_per_second:
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        for frame_number in random_frame_list:
            time_position_seconds = frame_number / fps
            vidcap.set(cv2.CAP_PROP_POS_MSEC, time_position_seconds * 1000)
            success, image = vidcap.read()

            if success:
                frame_filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
                """
                if image_crop_check:
                    image = crop_to_square(image, dimension_x, dimension_y)
                    if image is None:
                        print(f"Error: Failed to crop frame {frame_number}")
                        continue
                
                """
                cv2.imwrite(frame_filename, image)
                print(f"Frame {frame_number} extracted")
            else:
                print(f"Error extracting frame {frame_number}")
    else:
        frames_to_extract = int(0.3 * total_frames)
        frame_indices = random.sample(range(total_frames), frames_to_extract)

        for i, frame_index in enumerate(frame_indices):
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, image = vidcap.read()
            if not success:
                break
            if image_crop_check:
                image = crop_to_center(image)
                if image is None:
                    print(f"Error: Failed to crop frame {frame_index}")
                    continue
            image_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
            cv2.imwrite(image_path, image)
            print(f"Saved frame {i+1}/{frames_to_extract} as {image_path}")

    vidcap.release()


def main(video_path, output_path):
    extract_frames(video_path, output_path, True, True)

if __name__ == "__main__":
    video_path = "/Users/felixlu/Downloads/HYS Cello Sample Audition (480p60).mp4"
    output_path = "/Users/felixlu/Desktop/Evaluator_Videos"
    main(video_path, output_path)

# import shutil
# from google.colab import files

# # Replace 'your_folder' with the path to the folder you want to download
# folder_to_zip = 'posture_video_dataset'
# output_filename = 'posture_video_dataset.zip'

# # Zip the folder
# shutil.make_archive(output_filename.replace('.zip', ''), 'zip', folder_to_zip)

# # Download the zipped folder
# files.download(output_filename)


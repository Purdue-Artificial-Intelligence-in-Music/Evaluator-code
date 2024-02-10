#pip install pytube
import os
from pytube import YouTube
import random

yt_ids = [
    "3TR7YwNESt0",
    "GkT1NLMY7xQ",
    "b28_gkUgqoE",
    "GMyL9JTfDps",
    "iNef2eekeK0",
    "QPfE-FqhQq4",
    "P--7W1h4khQ",
    "N7te_FRdSgk",
    "0jSLgzeFmao",
    "IpCafmoaImc",
    "PR1Pzaqew9w",
    "SPh_q2n3JYM",
    "ww9_f6hxYKo",
    "REhp68K-6mw",
    "l9ufvevHbkE",
    "TiqUpTAwWgY",
    "CNDZfj3BaqA",
    "voJaSCe8UyM",
    "UWRZ4gi3HAM",
    "5IwWfy2RhQE",
    "VEaJzYRrrD8",
    "nAtHjkKGr4A",
    "MKsYR-9ZD9M",
    "07kcsRHipyw",
    "Q0OfnLEpty4",
    "6--mQjcImjM",
    "3PrFqTav5Ow",
    "WSheIMbyC5w",
    "uIqbE0Ylh9o",
    "ebMlp9Kc_bY",
    "OvXAT3_GGec",
    "LDq9sFUjwQk",
    "J_nJYlZHpuw",
    "wXq_IU_vvrI",
    "Z1rRodHDapI",
    "6GEkwyK_tns",
    "s92f3CW9IdQ",
    "MHhK6jG6U2I",
    "vI-A3KJSBpk",
    "ZNs5igA6tZQ",
    "JxhUFn5CT6A",
    "iXnGftY7k_k",
    "6XJCvBBQu78",
    "BvrGLXMvc_o",
    "t81IJJVi-Us",
    "7nzwswUUDto",
    "-ZMb-QiqDJY",
    "zGiFcVKafQM",
    "hC7XPGO45hk",
    "VdFnyZWJAgo",
    "eRpbaoIOJjI",
    "dfPeeHyfpP8",
    "2C7_o5mfE6k",
    "dKBVWebVakI",
    "gRy76nNP6CQ",
    "mGQLXRTl3Z0",
    "PDJ_QZAbGi4",
    "w7Kh0LscP3U",
    "1u3yHICR_BU",
    "sBZJCnxdmPc",
    "yC8ovVGeUfA",
    "DpvxiC0osbA",
    "MNB8H2p9Kk0",
    "5sXcGZ60NpI",
    "kG3CrRcMgMk",
    "dTmuOKUPsYI",
    "RDnLgY2-abc",
    "WotHO4X1kak",
    "I0LedcEaPL0",
    "7WeXhVxpTAw",
    "JW9-YhFMYSE",
    "8PXQeH0g6jI",
    "wHfhjkI2L9M",
    "4UI2fPRaxpg",
    "7_8yPpCjK_w",
    "aLTH6DVXAcU",
    "KtsdeqrcO2U",
    "iuvaHDIGP5U",
    "mmJim2h-byQ",
    "FR1UZKD238Y",
    "42ULINLSfrE",
    "A2d2DHxV4l0",
    "4jxmrdkyjCc",
    "09iOwYycY6w",
    "AT97pTC97tY",
    "j9Re7hWGh3Y",
    "Zz9nwbi9oqA",
    "j4CwTQmRxiY",
    "vWytPM5Wawk",
    "WSvQ7_N1y08",
    "llpPao0uM7U",
    "-qRn8UyHogA",
    "MbuRvmZy3Yw",
    "MaYRimw-mx8",
    "Ng8QfKTk1KA",
    "seWlG50y-so",
    "fLH9GyDbUjU",
    "bw2Etrk6cHg",
    "fJ7yh1E9S-k",
    "1X2l_1za1g0",
    "DsQX--nVPJQ",
    "qxiDk7x6gkM",
    "BJW31sWO5cE",
    "CrBCIFZImKE",
    "3qHdCTe0mIs",
    "4-TAbbQmCtg",
    "qtIrc7YUEvo",
    "A2KrKTMSoEU",
    "Jsu5p6iAApI",
    "xFlBZMiVMT0",
    "qtVf-KkpX1w",
    "DEmhfGTWt8g",
    "PCicM6i59_I",
    "poCw2CCrfzA",
    "7Z96EdnfzoQ",
    "Yi9Xdiq579o",
    "qzl9w7gME50",
    "Vp2lSAv__wE",
    "VlIcqDWmPkw",
    "gtr9N_qdHag",
    "rifOR2Q0cRk",
    "bhxNofWXTEA",
    "aHiqoSUV5MA",
    "DB3kIKJPdUA",
    "RV4Ob2Y9hvQ",
    "BObT8M5Mnus",
    "bNLswbcgbF4",
    "stgIDZN6e5Q",
    "eWTR-puxJyo",
    "Y1ZO2qfr0ko",
    "D_a1o8biIAk",
    "PpcSlOZY3Aw",
    "heFoLuid8-Y",
    "JQOC7Za4Pbw"
  ]


# Function to extract frames from a video

def extract_frames(youtube_video_id, output_folder, fps=1):
  
    # Download the YouTube video
    yt = YouTube(f"https://www.youtube.com/watch?v={youtube_video_id}")
    video = yt.streams.get_highest_resolution()
    video_path = video.download(output_path=output_folder)
    video_filename = os.path.basename(video_path)

    # Extract frames from the downloaded video
    extract_frames_from_video(video_path, output_folder)

    # Delete the downloaded video file
    os.remove(video_path)

# Grabs random unique integers within 0 to max number of frames per a video
def generate_unique_random_list(start, end, length):
    if length > (end - start + 1):
        raise ValueError("Length of list cannot exceed the range of integers.")
    return random.sample(range(start, end + 1), int(length))

import os
import cv2
from random import sample

# grabs random frames from downloaded video and stores them in a folder
def extract_frames_from_video(video_path, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    #Calculate total number of frames and a list of random integers from
    # 0 to total number of frames
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_to_extract = int(0.3 * total_frames)

    # Generate a list of unique random frame indices to extract
    frame_indices = sample(range(total_frames), frames_to_extract)

    for i, frame_index in enumerate(frame_indices):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = vidcap.read()
        if not success:
            break
        image_path = os.path.join(output_folder, f"frame_{frame_index}.jpg")
        cv2.imwrite(image_path, image)
        print(f"Saved frame {i+1}/{frames_to_extract} as {image_path}")

    vidcap.release()

for i in range(0, len(yt_ids)):
  extract_frames(len(yt_ids), "/content/CelloVids")


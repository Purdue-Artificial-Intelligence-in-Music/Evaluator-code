from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import backend
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import subprocess
import shlex
import json
from fastapi.staticfiles import StaticFiles
import shutil


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify list like ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # or restrict to ["POST", "GET"]
    allow_headers=["*"],  # or restrict to specific headers
)

class ImagePayload(BaseModel):
    image: str

class VideoPayload(BaseModel):
    video: str


# TODO: use ip address of your computer here (use ipconfig or ifconfig to look up)
server_ip = ""

@app.post("/upload")
async def process_image(payload: ImagePayload):
    base64_image = payload.image

    if base64_image.startswith('data:image'):
        base64_image = base64_image.split(',')[1]

    padding_needed = len(base64_image) % 4
    if padding_needed != 0:
        base64_image += '=' * (4 - padding_needed)

    image_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    points = backend.processFrame(image)

    pointsDummy = [ ("box bow top left", backend.Point2D(200, 200)),
                            ("box bow top right", backend.Point2D(250, 200)), 
                            ("box bow bottom right", backend.Point2D(250, 270)), 
                            ("box bow bottom left", backend.Point2D(200, 270)), 
                            
                            ("box string top left", backend.Point2D(100, 100)), 
                            ("box string top right", backend.Point2D(300, 100)), 
                            ("box string bottom right", backend.Point2D(300, 150)), 
                            ("box string bottom left", backend.Point2D(100, 150)),

                            ("supination", "Invalid")
                          ]


    response_data = {}

    #this logic depends on the data being the correct order (bow box, string box, hands, supination)
    if (len(points) == 9 or len(points) == 30):
        for point in points:
            if (point[0] != "supination"):
                response_data.update({point[0]: (point[1]).to_dict()})
            else:
                response_data.update({point[0]: point[1]})
    else:
        for point in pointsDummy:
            if (point[0] != "supination"):
                response_data.update({point[0]: (point[1]).to_dict()})
            else:
                response_data.update({point[0]: point[1]})


    '''
    #old logic for routing points
    if (len(points) == 9):
        response_data.update({points[5][0]: (points[5][1]).to_dict()})
        response_data = {
            points[0][0]: (points[0][1]).to_dict(),
            points[1][0]: (points[1][1]).to_dict(),
            points[2][0]: (points[2][1]).to_dict(),
            points[3][0]: (points[3][1]).to_dict(),
            points[4][0]: (points[4][1]).to_dict(),
            points[5][0]: (points[5][1]).to_dict(),
            points[6][0]: (points[6][1]).to_dict(),
            points[7][0]: (points[7][1]).to_dict(),

            points[8][0]: (points[8][1])
        }
    else:
                response_data = {
                    pointsDummy[0][0]: (pointsDummy[0][1]).to_dict(),
                    pointsDummy[1][0]: (pointsDummy[1][1]).to_dict(),
                    pointsDummy[2][0]: (pointsDummy[2][1]).to_dict(),
                    pointsDummy[3][0]: (pointsDummy[3][1]).to_dict(),
                    pointsDummy[4][0]: (pointsDummy[4][1]).to_dict(),
                    pointsDummy[5][0]: (pointsDummy[5][1]).to_dict(),
                    pointsDummy[6][0]: (pointsDummy[6][1]).to_dict(),
                    pointsDummy[7][0]: (pointsDummy[7][1]).to_dict(),

                    pointsDummy[8][0]: (pointsDummy[8][1])
                }
                '''

    return response_data


# @app.post("/send-video")
# async def upload_video(payload: VideoPayload):
#     print("Received video")

#     path = str(Path(__file__).parent.parent / "demo1.mp4")
#     print (path)
#     demo1 = path

#     try :
#         os.remove(demo1)
#     except FileNotFoundError:
#         pass

    
#     output_video = backend.videoFeed(payload.video)

#     print(demo1)
#     print(output_video)

#     #Convert the AVI file into Mp4 using 

#     try:
#         result = subprocess.run(
#         ['ffmpeg', '-i', output_video, '-vf', "transpose=2", demo1],
#         capture_output=True,
#         text=True
#         )
    
#         if result.returncode == 0:
#             output_video = demo1
#             print("Conversion completed")
#         else:
#             print("Conversion failed:")
#             print(result.stderr)
#     except FileNotFoundError:
#         print("ffmpeg not found. Make sure it's installed and in your PATH.")

#     # Following code based on stackoverflow page on getting resolution from ffmpeg
#     cmd = "ffprobe -v quiet -print_format json -show_streams"
#     args = shlex.split(cmd)
#     args.append(demo1)
#     # run the ffprobe process, decode stdout into utf-8 & convert to JSON
#     ffprobeOutput = subprocess.check_output(args).decode('utf-8')
#     ffprobeOutput = json.loads(ffprobeOutput)

#     # find height and width
#     height = ffprobeOutput['streams'][0]['height']
#     width = ffprobeOutput['streams'][0]['width']

#     # Encode the file into base64
#     with open(demo1, "rb") as f:
#         encoded = base64.b64encode(f.read()).decode("utf-8")
#         output_video = f"data:video/mp4;base64,{encoded}"

#     response_data = { "Video": output_video,
#                      "Height": height,
#                      "Width": width }


#     return response_data


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)  # create /static
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# base64 + return url
@app.post("/send-video")
async def upload_video(payload: VideoPayload):
    print("Received video")

    current_dir = Path(__file__).parent
    temp_video_path = str(current_dir / "temp_input.mp4")
    output_path = str(static_dir / "processed_video.mp4")  # save file to /static
    rotated_path = str(static_dir / "rotated_video.mp4")

    try:
        # clean old files
        for file_path in [temp_video_path, output_path, rotated_path]:
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass

        # parse base64 video data
        if "," in payload.video:
            video_data = base64.b64decode(payload.video.split(",")[1])
        else:
            video_data = base64.b64decode(payload.video)

        with open(temp_video_path, "wb") as f:
            f.write(video_data)

        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Failed to open input video file")
        cap.release()

        # call backend function, output saved to output_path
        backend.videoFeed(temp_video_path, output_path)
        print("Processed and saved:", output_path)

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise Exception("Output video missing or empty")

        # ffmpeg rotate counter-clockwise
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise Exception("FFmpeg not found. Please install and add to PATH.")

        result = subprocess.run(
            [ffmpeg_path, "-i", output_path, "-vf", "transpose=2", rotated_path],
            capture_output=True,
            text=True,
            timeout=30  # add timeout protection
        )

        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")

        # get video size
        cap = cv2.VideoCapture(rotated_path)
        if not cap.isOpened():
            raise Exception("Failed to open rotated video file")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # construct public/local access URL
        video_url = f"http://{server_ip}:8000/static/rotated_video.mp4"

        return {
            "Video": video_url,
            "Height": height,
            "Width": width
        }

    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
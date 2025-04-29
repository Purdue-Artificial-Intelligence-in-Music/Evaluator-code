import base64
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from . import backend

from .serializers import UploadedImageSerializer

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view

from django.http import JsonResponse

@method_decorator(csrf_exempt, name='dispatch')
class UploadImageView (APIView): 
    @csrf_exempt
    def post(self, request):
        print("received") # check the request data

        serializer = UploadedImageSerializer(data=request.data)
        if serializer.is_valid():
            base64_image = request.data.get('image')

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

            print("length of points " + str(len(points)))

            print("points " + str(points))
            response_data = {}

            if (len(points) == 9):
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
                
            print(response_data)

            return Response(response_data, status=201)
        else:
            return Response(serializer.errors, status=400)
        

@method_decorator(csrf_exempt, name='dispatch')
class UploadVideoView (APIView): 
    @csrf_exempt
    def post(self, request):

        print("Received video")

        path = str(Path(__file__).parent.parent / "demo1.mp4")
        demo1 = path

        try :
            os.remove(demo1)
        except FileNotFoundError:
            pass

        
        output_video = backend.videoFeed(request.data.get('Video'))

        #Convert the AVI file into Mp4 using 

        try:
            result = subprocess.run(
            ['ffmpeg', '-i', output_video, '-vf', "transpose=2", demo1],
            capture_output=True,
            text=True
            )
        
            if result.returncode == 0:
                output_video = demo1
            else:
                print("Conversion failed:")
                print(result.stderr)
        except FileNotFoundError:
            print("ffmpeg not found. Make sure it's installed and in your PATH.")

        # Encode the file into base64
        with open(demo1, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            output_video = f"data:video/mp4;base64,{encoded}"

        response_data = {"Video": output_video}


        return Response(response_data, status = 201)
    
    
@method_decorator(csrf_exempt, name='dispatch')
class FilePathVideo (APIView): 
    @csrf_exempt
    def post(self, request):
        print("Received video")

        video = request.data.get('Video')

        full_path = None

        # This scans the downloads folder for the path of the video

        for root, _, files in os.walk(os.path.expanduser("~/Downloads")):
            if video in files:
                full_path = os.path.join(root, video)

        print(full_path)

        response = {"Video":full_path}

        return Response(response, status = 201)

        


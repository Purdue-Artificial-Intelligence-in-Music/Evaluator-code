import base64
import cv2
import numpy as np
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
            pointsDummy = [ ("dum1", backend.Point2D(0, 0)),
                            ("dum2", backend.Point2D(30, 30)), 
                            ("dum3", backend.Point2D(60, 60)), 
                            ("dum4", backend.Point2D(90, 90)), 
                            ("dum5", backend.Point2D(120, 120)), 
                            ("dum6", backend.Point2D(150, 150)), 
                            ("dum7", backend.Point2D(180, 180)), 
                            ("dum8", backend.Point2D(210, 210)),
                          ]

            print("length of points " + str(len(points)))

            print("points " + str(points))
            response_data = {}

            if (len(points) == 8):
                response_data = {
                    points[0][0]: (points[0][1]).to_dict(),
                    points[1][0]: (points[1][1]).to_dict(),
                    points[2][0]: (points[2][1]).to_dict(),
                    points[3][0]: (points[3][1]).to_dict(),
                    points[4][0]: (points[4][1]).to_dict(),
                    points[5][0]: (points[5][1]).to_dict(),
                    points[6][0]: (points[6][1]).to_dict(),
                    points[7][0]: (points[7][1]).to_dict(),
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
                }
                
            print(response_data)

            return Response(response_data, status=201)
        else:
            return Response(serializer.errors, status=400)
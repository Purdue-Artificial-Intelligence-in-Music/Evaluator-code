from fastapi import FastAPI, Request
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import backend
from fastapi.middleware.cors import CORSMiddleware

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

    return response_data

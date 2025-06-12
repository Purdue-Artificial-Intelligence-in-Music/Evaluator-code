from fastapi import FastAPI, Request, UploadFile, File, HTTPException
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
from fastapi.responses import JSONResponse
import shutil
import uuid


def main():
    STATIC_DIR = "static"
    filename = "e8e28d0d-8f9e-4fa4-9ca1-d085bfb97860.mp4"
    save_path = os.path.join(STATIC_DIR, filename)
    print("Current working directory:", os.getcwd())
    print("save_path:", save_path)
    # Step 2: get video size
    cap = cv2.VideoCapture(save_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cap.release()

    # Step 3: process video
    processed_path = os.path.join(STATIC_DIR, "test_processed_video.mp4")
    backend.videoFeed(save_path, processed_path)
    print("Processed and saved:", processed_path)


if __name__ == "__main__":
    main()

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import statistics
import math

# Load a model
model = YOLO("nano_best.pt")
result = model.export(format='tflite', device='mps', nms=True)
print(result)
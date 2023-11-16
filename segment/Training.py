#The lines below import the specificly trained data from Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="tw1RlN5Mqqxps1xAAsMB")
project = rf.workspace("purdue-aim").project("bow-and-strings")
dataset = project.version(2).download("yolov5")

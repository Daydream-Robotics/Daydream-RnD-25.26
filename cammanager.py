from picamera2 import Picamera2
from libcamera import controls
import numpy as np

picam2 = Picamera2()

picam2.video_configuration.main.size = (640, 640)
picam2.video_configuration.controls.FrameRate = 120
picam2.configure("video")

picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})

picam2.start()


def getCamNumpy():
    return picam2.capture_array("main")

def getCamPIL():
    return picam2.capture_image("main")

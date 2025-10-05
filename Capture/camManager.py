from picamera2 import Picamera2
import numpy as np

class CamManager:

    def __init__(self):
        self.picam2 = Picamera2()

        self.picam2.video_configuration.main.size = (640, 640)
        self.picam2.video_configuration.controls.FrameRate = 120
        self.picam2.configure("video")

        self.picam2.start()


    def getCamNumpy(self):
        return self.picam2.capture_array("main")

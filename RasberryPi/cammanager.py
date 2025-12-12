# UCF7 Daydream
# Purpose: Configure camera and return live images
from picamera2 import Picamera2
from libcamera import controls
import numpy as np

picam2 = Picamera2() # creates a camera object
picam2.video_configuration.main.size = (640, 640) # sets the image dimensions
picam2.video_configuration.controls.FrameRate = 120 # sets the video frame rate
picam2.configure("video") # sets camera mode (allows for faster frame retrieval)
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # sets autofocus mode
picam2.start() # starts camera object

# this just returns the most recent frame
def getCamPIL():
    return picam2.capture_image("main")


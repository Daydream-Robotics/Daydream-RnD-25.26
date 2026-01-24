# UCF7 Daydream
# Purpose: Configure camera and return live images
from picamera2 import Picamera2
from libcamera import controls, Transform
import numpy as np

picam2 = Picamera2() # creates a camera object
config = picam2.create_video_configuration(
    main={"format": "RGB888"},
    # main={"size": (640, 640), "format": "RGB888"},
    transform=Transform.Rot90,  # Rotates hardware read-out
    controls={"FrameRate": 120}
)
picam2.configure(config) # sets camera configuration
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous}) # sets autofocus mode
picam2.start() # starts camera object

# this just returns the most recent frame
def getCamPIL():
    return picam2.capture_image("main")


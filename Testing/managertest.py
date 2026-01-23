from camManager import CamManager
import time

cam_manager = CamManager()

time_between_print = 1

start_time = time.time()
capturing = True
frames = 0
while capturing:
    cam_manager.getCamNumpy()
    frames += 1

    if time.time() - start_time >= time_between_print:
        print(frames/time_between_print)
        frames = 0
        start_time = time.time()


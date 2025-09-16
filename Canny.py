import cv2 as cv
import numpy as np
from picamera2 import Picamera2

# --- Camera setup ---
picam2 = Picamera2()
# keep resolution small for Zero (e.g., 320x240)
config = picam2.create_preview_configuration(
    main={"size": (320, 240), "format": "YUV420"}
)
picam2.configure(config)
picam2.start()

# --- ROI parameters (crop around center for now) ---
ROI_SIZE = 100
def get_roi_coords(img_w, img_h, cx=None, cy=None, size=ROI_SIZE):
    if cx is None: cx = img_w // 2
    if cy is None: cy = img_h // 2
    x0 = max(0, cx - size//2)
    y0 = max(0, cy - size//2)
    x1 = min(img_w, x0 + size)
    y1 = min(img_h, y0 + size)
    return x0, y0, x1, y1

while True:
    # capture frame
    req = picam2.capture_request()
    yuv = req.make_array("main")      # YUV420 image
    req.release()

    H, W = 240, 320
    Y = yuv[:H, :W]                   # extract Y plane (grayscale)

    # crop ROI in middle for demo
    x0, y0, x1, y1 = get_roi_coords(W, H)
    roi = Y[y0:y1, x0:x1]

    # run Canny
    edges = cv.Canny(roi, 60, 120, L2gradient=False)

    # visualize
    vis = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
    vis[edges > 0] = (0, 0, 255)  # draw edges in red

    cv.imshow("ROI edges", vis)
    if cv.waitKey(1) & 0xFF == 27:  # press Esc to quit
        break

cv.destroyAllWindows()

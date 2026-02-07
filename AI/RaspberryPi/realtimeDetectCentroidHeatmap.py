# import tensorflow as tf
import tflite_runtime.interpreter as tflite

import numpy as np
from PIL import Image, ImageEnhance
import time, cammanager
import cv2
# from tensorflow.lite.python.interpreter import load_delegate
    

       ###########
    ###           ###
  ##                 ##
 #                     #
#########################
#    MODEL GRAVEYARD    #
# --------------------- #
# best_model.tflite     #
# INSANE_model.tflite   #
# BGObjectsBest.tflite  #
# HigherNeg1.tflite     #
# NoisyModel.tflite     #
# NoisyModel2.tflite    #
# INSANE_model.tflite   #
#########################


class CentroidDetector:
    
    def __init__(self, model_path, class_names=None):
        # print("Input Type:", self.input_details[0]['dtype'])
        # Load TFLite Model
        # self.interpreter = tf.lite.Interpreter(
        #     model_path=model_path, 
        #     num_threads=3
        # )

        self.interpreter = tflite.Interpreter(
            model_path=model_path, 
            num_threads=3, 
            experimental_delegates=[
                # tflite.load_delegate('libxnnpack_delegate.so')
            ]
        )
        
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        # self.inpu
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_size = tuple(self.input_details[0]['shape'][1:3])
        #self.input_size = (256, 256)

        # Class names
        self.class_names = class_names or [
            "RedBall", "BlueBall"
        ]
        
        print("✅ Model Loaded")
        print("Output shapes:", [o["shape"] for o in self.output_details])
    
    # --------------------------
    # Preprocess
    # --------------------------
    def _preprocess(self, pil_img):
        img = pil_img.convert("RGB").resize(self.input_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        # arr = np.array(img, dtype=np.uint8)
        arr = np.expand_dims(arr, axis=0)
        return arr
    
    # --------------------------
    # Predict
    # --------------------------
    def _predict(self, pil_img):
        # Preprocess image
        input_data = self._preprocess(pil_img)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        t0 = time.perf_counter()
        self.interpreter.invoke()
        t_ms = (time.perf_counter() - t0) * 1000
        # print(f"Invoke {t_ms} ms")
       
        # Capture output
        p8_hm, p8_off, p16_hm, p16_off = [
           self.interpreter.get_tensor(o["index"]) for o in self.output_details
        ]

        # print(np.shape(p8_hm))

        # # # Capture all outputs first
        # outputs = [self.interpreter.get_tensor(o["index"]) for o in self.output_details]
        #
        # # Map outputs by shape to ensure correct assignment
        # # We assume 4 classes (depth=4) and 2 offsets (depth=2)
        # p8_hm = p16_hm = p8_off = p16_off = None
        #
        # for out in outputs:
        #     shape = out.shape
        #     h, w, c = shape[1], shape[2], shape[3]
        #
        # #    # Stride 8 (Larger spatial dim, e.g., 64x64)
        #     if h == self.input_size[0] // 16:
        #         if c == 4: p8_hm = out     # Heatmap
        #         elif c == 2: p8_off = out  # Offsets
        #
        #     # Stride 16 (Smaller spatial dim, e.g., 32x32)
        #     elif h == self.input_size[0] // 32:
        #         if c == 4: p16_hm = out    # Heatmap
        #         elif c == 2: p16_off = out # Offsets

        # Safety check
        if p8_hm is None or p8_off is None:
            raise ValueError("Could not map TFLite outputs to Heatmaps/Offsets. Check tensor shapes.")
        
        return {"p8": p8_hm, "p16": p16_hm, "off8": p8_off, "off16": p16_off, "time_ms": t_ms}
    
    # --------------------------
    # Decode heatmaps -> detections
    # --------------------------
    def _decode(self, outputs, conf_thresh=0.3, top_k=5, use_multiscale=True):
        def decode_single(hm, off):
            hm = hm[0]  # remove batch
            off = off[0]

            # Softmax with stabilization
            hm = np.exp(hm - np.max(hm, axis=-1, keepdims=True))
            hm /= np.sum(hm, axis=-1, keepdims=True)

            h, w, c = hm.shape
            detections = []
            for cls in range(c):
                if cls > 1:
                    continue

                classwise_map = hm[:, :, cls]
                for _ in range(top_k):
                    idx = np.unravel_index(np.argmax(classwise_map), classwise_map.shape)
                    conf = classwise_map[idx]
                    if conf < conf_thresh:
                        break
                    y, x = idx
                    ox, oy = off[y, x]
                    cx = (x + ox) / w
                    cy = (y + oy) / h
                    detections.append((cls, cx, cy, float(conf)))
                    classwise_map[y, x] = 0  # suppress this peak
            return detections

        dets_p8 = decode_single(outputs["p8"], outputs["off8"])
        dets_p16 = decode_single(outputs["p16"], outputs["off16"]) if use_multiscale else []
        all_dets = dets_p8 + dets_p16
        return all_dets
    
    # --------------------------
    # Centroid NMS
    # --------------------------
    def _centroid_nms(self, dets, dist_thresh=0.05):
        keep = []
        dets = sorted(dets, key=lambda x: x[3], reverse=True)
        while dets:
            best = dets.pop(0)
            keep.append(best)
            dets = [
                d for d in dets
                if d[0] != best[0] or np.hypot(d[1] - best[1], d[2] - best[2]) > dist_thresh
            ]
        return keep
    
    # --------------------------
    # Inference Wrapper
    # --------------------------
    def infer(self, conf_thresh=0.3, show_preview=False):
        t0 = time.perf_counter()
        input_image = cammanager.getCamPIL()
        t0_ms = (time.perf_counter() - t0) * 1000
        # print(f"Img gra: {t0_ms:.1f} ms")

        # enhancer = ImageEnhance.Color(input_image)
        # saturation_factor = 1.5
        # input_image = enhancer.enhance(saturation_factor)

        # enhancer = ImageEnhance.Brightness(input_image)
        # bright_image = enhancer.enhance(1.5)

        
        t1 = time.perf_counter()
        x = self._predict(input_image)
        t1_ms = (time.perf_counter() - t1) * 1000
        # print(f"Predict: {t1_ms:.1f} ms")

        t2 = time.perf_counter()
        x = self._decode(x, conf_thresh=conf_thresh, top_k=5, use_multiscale=True)
        t2_ms = (time.perf_counter() - t2) * 1000
        # print(f"Decode: {t2_ms:.1f} ms")

        t3 = time.perf_counter()
        results = self._centroid_nms(x, dist_thresh=0.15)
        t3_ms = (time.perf_counter() - t3) * 1000
        # print(f"NMS: {t3_ms:.1f} ms")
        
        t4 = time.perf_counter()
        if show_preview:
            # Convert PIL (RGB) to OpenCV (BGR)
            frame = np.array(input_image) 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Get dimensions for scaling normalized coords back to pixels
            img_h, img_w = frame.shape[:2]

            for det in results:
                # Unpack tuple: (class_name, cx, cy, conf)
                label, cx, cy, conf = det 
                label = self.class_names[label]
                
                # Convert normalized (0-1) to pixels
                x = int(cx * img_w)
                y = int(cy * img_h)
                
                # Draw Circle (Green)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                
                # Draw Text (White with Black Outline for readability)
                text = f"{label}: {conf:.2f}"
                cv2.putText(frame, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Show the window
            cv2.imshow("Detector Preview", frame)
            
            # Required to update the window (1ms delay)
            cv2.waitKey(1)
            t4_ms = (time.perf_counter() - t4) * 1000
            # print(f"Preview: {t4_ms:.1f} ms")

        t_ms= (time.perf_counter() - t0) * 1000
        # print(f"🔹 Inference + decode: {t_ms:.1f} ms ({1000 / t_ms:.1f} FPS)")
        return results

# MODEL = 'NullModel1.tflite'
MODEL = 'NoisyModel2.tflite'

# MODEL = 'best_model.tflite'
MODEL = '/home/adam/Daydream-RnD-25.26/AI/RaspberryPi/TfliteModels/FGThresh1.tflite'
# MODEL = 'HigherNeg1.tflite'
detector = CentroidDetector(MODEL)

def step(conf_threshold=0.6, show_preview=False):
    detections = detector.infer(conf_threshold, show_preview)

    objects = []

    if detections:
        # sort detections by y
        detections.sort(key=lambda x: x[2], reverse=True)
        
        for detection in detections:
            object = {
                "class_id": detection[0],
                "x": detection[1],
                "y": detection[2],
                "conf": detection[3],
            }
            objects.append(object)

    return objects

while True:
    objects = step(show_preview=True)

    # Print detection summary
    num_objects = len(objects)
    print(f"\nDetected {num_objects} objects:")
    for i, object in enumerate(objects):
        # class_id = int(object["class_id"])
        conf = object["conf"]
        x = object["x"]
        y = object["y"]
        # class_name = detector.class_names[class_id]
        class_name = object["class_id"]
        print(f"  {i+1}. {class_name}: {conf:.2%} confidence at [{x:.0f}, {y:.0f}]") 

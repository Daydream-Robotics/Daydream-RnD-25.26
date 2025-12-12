import tensorflow as tf
import numpy as np
from PIL import Image
import time, cammanager


class CentroidDetector:
    
    def __init__(self, model_path, class_names=None):
        # Load TFLite Model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_size = tuple(self.input_details[0]['shape'][1:3])
        
        # Class names
        self.class_names = class_names or [
            "RedBall", "BlueBall", "LongGoal", "MiddleGoalTop", "MiddleGoalBottom"
        ]
        
        print("✅ Model Loaded")
        print("Output shapes:", [o["shape"] for o in self.output_details])
    
    # --------------------------
    # Preprocess
    # --------------------------
    def _preprocess(self, pil_img):
        img = pil_img.convert("RGB").resize(self.input_size)
        arr = np.array(img, dtype=np.float32) / 255.0
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
        
        # Capture output
        p8_hm, p16_hm, p8_off, p16_off = [
            self.interpreter.get_tensor(o["index"]) for o in self.output_details
        ]
        
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
                    detections.append((self.class_names[cls], cx, cy, float(conf)))
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
    def infer(self):
        t0 = time.perf_counter()
        input_image = cammanager.getCamPIL()
        
        x = self._predict(input_image)
        x = self._decode(x, conf_thresh=0.3, top_k=5, use_multiscale=True)
        x = self._centroid_nms(x)
        
        t_ms= (time.perf_counter() - t0) * 1000
        print(f"🔹 Inference + decode: {t_ms:.1f} ms ({1000 / t_ms:.1f} FPS)")
        return x

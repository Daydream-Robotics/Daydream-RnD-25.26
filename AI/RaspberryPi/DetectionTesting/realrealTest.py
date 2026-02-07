# import tensorflow as tf
from ai_edge_litert.tensor_buffer import TensorBuffer
from ai_edge_litert.compiled_model import CompiledModel

import numpy as np
from PIL import Image, ImageEnhance
import time, cammanager
import cv2

class CentroidDetector:
    # Complete fixes for INT8 quantized model

    def __init__(self, model_path, class_names=None):
        # ⚠️ CRITICAL: Set to 256 based on your export logs.
        self.input_size = (256, 256) 

        self.model = CompiledModel.from_file(model_path)
        
        self.signature_name = list(self.model.get_signature_list().keys())[0]
        signature = self.model.get_signature_list()[self.signature_name]

        self.input_name = signature["inputs"][0]
        self.output_names = signature["outputs"]

        # 1. Create Input Buffer once (Correctly typed by the model)
        self.in_buffer = self.model.create_input_buffer_by_name(self.signature_name, self.input_name)

        # 2. Create Output Buffers
        self.out_buffers = {
            name: self.model.create_output_buffer_by_name(self.signature_name, name)
            for name in self.output_names
        }

        # Expected output shapes for Stride 4 (64x64) and Stride 8 (32x32) at 256 input
        self.output_shapes = [
            (1, 64, 64, 4),
            (1, 64, 64, 2),
            (1, 32, 32, 4),
            (1, 32, 32, 2),
        ]

        if len(self.output_names) != len(self.output_shapes):
            raise ValueError(f"Expected {len(self.output_shapes)} outputs, got {len(self.output_names)}: {self.output_names}")

        self.class_names = class_names or [
            "RedBall", "BlueBall", "LowLeg", "HighLeg"
        ]
        
        print("✅ Model Loaded")
        print("LiteRT output names order:", self.output_names)
        print("Using output_shapes order:", self.output_shapes)

    def _preprocess(self, pil_img):
        """
        Preprocess PIL image for model inference.
        Convert to float32 normalized [0, 1] for input (model quantizes internally).
        """
        # Resize to model input size
        img = pil_img.convert("RGB").resize(self.input_size)

        # Convert to float32 normalized [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)

        # Ensure C-contiguous memory layout
        arr = np.ascontiguousarray(arr)

        return arr

    # --------------------------
    # Predict
    # --------------------------
    def _predict(self, pil_img):
        # Preprocess image
        input_data = self._preprocess(pil_img)

        # Load data into the existing buffer using .load() not .write()
        try:
            self.in_buffer.load(input_data)
        except ValueError as e:
            print(f"❌ Buffer Fill Failed!")
            print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
            raise e

        # Run inference
        t0 = time.perf_counter()
        
        # Pass the pre-filled self.in_buffer
        self.model.run_by_name(self.signature_name, {self.input_name: self.in_buffer}, self.out_buffers)
        
        t_ms = (time.perf_counter() - t0) * 1000

        # Capture outputs
        outputs = []
        for name, shape in zip(self.output_names, self.output_shapes):
            outputs.append(self._read_buffer(self.out_buffers[name], shape))

        # Map outputs by shape
        p8_hm = p16_hm = p8_off = p16_off = None

        for out in outputs:
            shape = out.shape
            h, w, c = shape[1], shape[2], shape[3]
            
            # Stride 8 equivalent (Large map 64x64)
            if h == 64: 
                if c == 4: p8_hm = out
                elif c == 2: p8_off = out
            
            # Stride 16 equivalent (Small map 32x32)
            elif h == 32:
                if c == 4: p16_hm = out
                elif c == 2: p16_off = out

        return {"p8": p8_hm, "p16": p16_hm, "off8": p8_off, "off16": p16_off, "time_ms": t_ms}

    def _read_buffer(self, buf, shape, dtype=np.int8):
        """
        Read buffer and dequantize if needed.
        For INT8 models, read as int8 then convert to float32.
        """
        numel = int(np.prod(shape))
        
        # Read as int8 (quantized)
        data = buf.read(numel, np.int8).reshape(shape)
        
        # Dequantize: int8 -> float32
        # Standard dequantization: float_value = (int_value - zero_point) * scale
        # For typical quantization: scale ≈ 0.003921 (1/255), zero_point = 0
        data_float = data.astype(np.float32)
        
        # Apply dequantization (adjust these values if your model uses different quantization)
        # Common approach: map [-128, 127] to approximately [-0.5, 0.5] or similar range
        data_float = data_float / 128.0  # Simple scaling, adjust if needed
        
        return data_float
        
    # --------------------------
    # Predict
    # --------------------------
    def _predict(self, pil_img):
        # Preprocess image
        input_data = self._preprocess(pil_img)

        # Load data into the existing buffer
        # FIX: 'write' expects the numpy array itself, not bytes
        try:
            self.in_buffer.write(input_data)
        except Exception as e:
            print(f"❌ Buffer Fill Failed!")
            print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")
            # print(f"Buffer available methods: {dir(self.in_buffer)}")
            raise e

        # Run inference
        t0 = time.perf_counter()
        
        # Pass the pre-filled self.in_buffer
        self.model.run_by_name(self.signature_name, {self.input_name: self.in_buffer}, self.out_buffers)
        
        t_ms = (time.perf_counter() - t0) * 1000

        # Capture outputs
        outputs = []
        for name, shape in zip(self.output_names, self.output_shapes):
            outputs.append(self._read_buffer(self.out_buffers[name], shape, np.float32))

        # Map outputs by shape
        p8_hm = p16_hm = p8_off = p16_off = None

        for out in outputs:
            shape = out.shape
            h, w, c = shape[1], shape[2], shape[3]
            
            # Stride 8 equivalent (Large map 64x64)
            if h == 64: 
                if c == 4: p8_hm = out
                elif c == 2: p8_off = out
            
            # Stride 16 equivalent (Small map 32x32)
            elif h == 32:
                if c == 4: p16_hm = out
                elif c == 2: p16_off = out

        return {"p8": p8_hm, "p16": p16_hm, "off8": p8_off, "off16": p16_off, "time_ms": t_ms}
    
    # --------------------------
    # Decode heatmaps -> detections
    # --------------------------
    def _decode(self, outputs, conf_thresh=0.3, top_k=5, use_multiscale=True):
        def decode_single(hm, off):
            hm = hm[0]  # remove batch
            off = off[0]

            # Softmax with stabilization
            #m = np.exp(hm - np.max(hm, axis=-1, keepdims=True))
            #hm /= np.sum(hm, axis=-1, keepdims=True)

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
    def infer(self, conf_thresh=0.3, show_preview=False):
        t0 = time.perf_counter()
        input_image = cammanager.getCamPIL()
        
        t1 = time.perf_counter()
        x = self._predict(input_image)
        
        t2 = time.perf_counter()
        x = self._decode(x, conf_thresh=conf_thresh, top_k=5, use_multiscale=True)
        
        t3 = time.perf_counter()
        results = self._centroid_nms(x)
        #results = x
        
        t4 = time.perf_counter()
        if show_preview:
            # Convert PIL (RGB) to OpenCV (BGR)
            frame = np.array(input_image) 
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            img_h, img_w = frame.shape[:2]

            for det in results:
                # Unpack tuple: (class_name, cx, cy, conf)
                label, cx, cy, conf = det 
                
                x = int(cx * img_w)
                y = int(cy * img_h)
                
                # Draw Circle (Green)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
                
                # Draw Text
                text = f"{label}: {conf:.2f}"
                cv2.putText(frame, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                cv2.putText(frame, text, (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow("Detector Preview", frame)
            cv2.waitKey(1)

        t_ms= (time.perf_counter() - t0) * 1000
        print(f"🔹 Inference + decode: {t_ms:.1f} ms ({1000 / t_ms:.1f} FPS)")
        return results


MODEL = 'WM1_INT8.tflite'
#MODEL = 'best_float16.tflite'
detector = CentroidDetector(MODEL)

def step(conf_threshold=0.3):
    detections = detector.infer(conf_threshold, show_preview=True)

    objects = []

    if detections:
        detections.sort(key=lambda x: x[3], reverse=True)
        
        for detection in detections:
            object = {
                "class_id": detection[0],
                "conf": detection[3],
                "x": detection[1],
                "y": detection[2]
            }
            objects.append(object)

    return objects

while True:
    objects = step(0)

    # Print detection summary
    num_objects = len(objects)
    print(f"\nDetected {num_objects} objects:")
    for i, object in enumerate(objects):
        conf = object["conf"]
        x = object["x"]
        y = object["y"]
        class_name = object["class_id"]
        print(f"  {i+1}. {class_name}: {conf:.2%} confidence at [{x:.2f}, {y:.2f}]")

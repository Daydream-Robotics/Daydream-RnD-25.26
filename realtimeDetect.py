import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import cv2
import random

import cammanager
import time

DISPLAY_DETECTIONS = False

class Detector:

    def __init__(self, model_path, class_names=None):
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape'][1:3]
        
        # COCO class names (default)
        self.class_names = class_names or [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Generate random colors for each class
        random.seed(42)
        self.colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                       for _ in range(len(self.class_names))]
        
    def predict(self, input_image, conf_threshold=0.25):
        # Preprocess image
        input_data = self._preprocess(input_image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Post-process detections
        detections = self._postprocess(output_data[0], conf_threshold)
        
        return detections
    
    def _preprocess(self, input_image):
        # # Load and resize image
        img = input_image.convert('RGB')
        self.original_size = img.size  # (width, height)
        img_resized = img.resize(self.input_shape)
        
        # # Convert to numpy array and normalize
        input_data = np.array(img_resized, dtype=np.float32)
        input_data = input_data / 255.0  # Normalize to [0, 1]
        
        # # Add batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data

    def _postprocess(self, output, conf_threshold):
        # YOLOv8 output format: [batch, 84, 8400] for COCO
        # 84 = 4 bbox coords + 80 class scores
        
        # Transpose to [8400, 84]
        predictions = output.transpose()
        
        # Extract boxes and scores
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
          
        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        # Filter by confidence
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        self.confidences = confidences[mask]
        self.class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return None
              
        # YOLOv8 format: [x_center, y_center, width, height]
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Check if coordinates are normalized (0-1 range) or in pixel values
        # If max value is <= 1.5, assume normalized coordinates
        if boxes.max() <= 1.5:
            # If coordinates are normalized (0-1), scale directly to original image size
            x_center *= self.original_size[0]
            y_center *= self.original_size[1]
            width *= self.original_size[0]
            height *= self.original_size[1]
            
        else:
            # Coordinates are in model input dimensions, scale to original
            scale_x = self.original_size[0] / self.input_shape[1]
            scale_y = self.original_size[1] / self.input_shape[0]
            
            x_center *= scale_x
            y_center *= scale_y
            width *= scale_x
            height *= scale_y
        
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

        detections = np.stack([self.class_ids, self.confidences, self.x_center, self.y_center], axis=0)
        return detections

    def _create_boxes(self):

        # Converts to corner format
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        coords = np.stack([self.x_center, self.y_center, self.width, self.height], axis=1)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            self.confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return {
                'boxes': boxes[indices],
                'scores': self.confidences[indices],
                'class_ids': self.class_ids[indices],
                'coords': coords[indices]
            }
        else:
            return {'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([]), 'coords': coords[indices]}
    

    def draw_detections(self, input_image, detections, thickness=3, font_scale=0.6):
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: PIL Image or numpy array
            detections: dict with 'boxes', 'scores', 'class_ids'
            thickness: box line thickness
            font_scale: font size scale
            
        Returns:
            PIL Image with drawn boxes
        """

        # Convert PIL to numpy if needed
        if isinstance(input_image, Image.Image):
            image_array = np.array(input_image)
        else:
            image_array = input_image.copy()

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        boxes = detections['boxes']
        scores = detections['scores']
        class_ids = detections['class_ids']
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color for this class
            color = self.colors[int(class_id)]
            
            # Draw rectangle
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label
            class_name = self.class_names[int(class_id)]
            label = f'{class_name} {score:.2f}'
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=2
            )
            
            # Draw label background
            cv2.rectangle(
                img_bgr,
                (x1, y1 - label_height - baseline - 10),
                (x1 + label_width + 10, y1),
                color,
                -1  # Fill
            )
            
            # Draw label text
            cv2.putText(
                img_bgr,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )
        
        # Convert back to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(img_rgb)
    
MODEL = 'yolov8n_saved_model/yolov8n_float16.tflite'
detector = Detector(MODEL)


def step(conf_threshold):
    input_image = cammanager.getCamPIL()

    detections = detector.predict(
        input_image=input_image,
        conf_threshold=conf_threshold
    )

    return detections




# # Usage Example
# if __name__ == "__main__":
#     # Initialize detector
#     ft = time.time()

#     try:
#         while True:
#             st = ft
#             # Run detection and save visualized result
#             input_image = cammanager.getCamPIL()

#             detections = detector.predict(
#                 input_image=input_image, 
#                 conf_threshold=0.25,
#                 iou_threshold=0.45
#             )

#             # annotated_img = detector.draw_detections(input_image, detections)
            
#             # Save image
#             # annotated_img.save(output_path)
#             # print(f"Saved annotated image to: {output_path}")
            
#             # Print detection summary
#             num_detections = len(detections['boxes'])
#             print(f"\nDetected {num_detections} objects:")
#             for i, (box, score, class_id) in enumerate(zip(
#                 detections['boxes'], 
#                 detections['scores'], 
#                 detections['class_ids']
#             )):
#                 class_name = detector.class_names[int(class_id)]
#                 print(f"  {i+1}. {class_name}: {score:.2%} confidence at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]")

        
#             # annotated_img_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)

#             # cv2.imshow("img", annotated_img_cv)
#             # cv2.waitKey(25)

#             ft = time.time()

#             fps = 1 / (ft - st)
#             print(f"fps: {fps}")

#     except KeyboardInterrupt:
#         print("Stopped by User")


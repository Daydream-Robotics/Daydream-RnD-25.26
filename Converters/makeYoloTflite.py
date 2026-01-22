from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.export(format='tflite', imgsz=256)
# model.export(format='tflite', int8=True)

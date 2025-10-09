from ultralytics import YOLO

model = YOLO('yolov8m.pt')
model.export(format='tflite')
model.export(format='tflite', int8=True)

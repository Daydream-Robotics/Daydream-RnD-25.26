from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Export directly to TFLite
# imgsz=320 sets the input size
# int8=True makes it faster on Raspberry Pi (optional, remove if accuracy drops)
model.export(format="tflite", imgsz=320)

import tensorflow as tf
import sys
import os

model_name = sys.argv[1]

# Load Keras model
model = tf.keras.models.load_model(model_name, compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
base_name = os.path.basename(model_name)
base_name = os.path.splitext(base_name)[0]

with open(base_name + ".tflite", 'wb') as f:
    f.write(tflite_model)

print("done")

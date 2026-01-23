import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model('best_model.keras', compile=False)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# Save
with open('best_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("done")

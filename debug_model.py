import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path

MODEL_PATH = "models/retinal_model.tflite"
IMAGE_PATH = "data/uploads/upload_20260621_225333.jpg"  # change this

CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()
out = interpreter.get_output_details()
print("Input shape:", inp[0]['shape'])
print("Input dtype:", inp[0]['dtype'])

# Load and preprocess image
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))
img = img.astype(np.float32)
img = (img / 127.5) - 1.0          # MobileNetV2 normalization
img = np.expand_dims(img, axis=0)  # shape [1,224,224,3]

# Run inference
interpreter.set_tensor(inp[0]['index'], img)
interpreter.invoke()
scores = interpreter.get_tensor(out[0]['index'])[0]

print("\nRaw scores:", scores)
for cls, score in zip(CLASSES, scores):
    print(f"  {cls}: {score*100:.1f}%")
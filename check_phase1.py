import tensorflow as tf
import numpy as np

CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

m = tf.keras.models.load_model("training/checkpoints/phase1_best.keras")
p = m.predict(np.random.rand(4, 224, 224, 3).astype("float32"), verbose=0)

for i, c in enumerate(CLASSES):
    print(f"  {c}: {p[:, i].mean()*100:.1f}%")
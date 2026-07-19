# paste in a new file: check_model.py
import tensorflow as tf
import numpy as np

CLASSES = ["Normal", "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD"]

for ckpt in ["training/checkpoints/phase2v2_best.keras",
             "training/checkpoints/phase2v2_final.keras"]:
    print(f"\n--- {ckpt} ---")
    model = tf.keras.models.load_model(ckpt)
    # random noise input
    dummy = np.random.rand(4, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy, verbose=0)
    print("Mean predictions per class:")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls}: {preds[:, i].mean()*100:.1f}%")
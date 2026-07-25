import numpy as np
import h5py
import os

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import matplotlib.pyplot as plt

# ======================== LOAD SAVED MODEL ========================
# Load the trained ResNet20 model (saved after 40 epochs of training)
model = keras.models.load_model('resnet20_turntable/model.keras')

# ======================== LOAD TEST DATA ==========================
# Load turntable test set — 645 unseen images the model was NOT trained on
# Each image is a 96x96 grayscale sonar patch of a single debris object
with h5py.File('data_down/marine-debris-turntable-classification-object_classes-platform-96x96.hdf5', 'r') as f:
    x_test = f['x_test'][:]    # shape: (645, 96, 96, 1), values: 0.0 to 1.0
    y_test = f['y_test'][:]    # shape: (645,), integer labels 0-11
    class_names = [c.decode() if isinstance(c, bytes) else str(c) for c in f['class_names'][:]]

# ======================== PREDICT & VISUALIZE =====================
# Pick 10 random test images (different each run)
np.random.seed(None)
indices = np.random.choice(len(x_test), 10, replace=False)

fig, axes = plt.subplots(2, 5, figsize=(16, 7))
fig.suptitle('ResNet20 Classification — Turntable Sonar Patches', fontsize=14)

for i, idx in enumerate(indices):
    ax = axes[i // 5, i % 5]
    img = x_test[idx]
    true_label = int(y_test[idx])

    # Feed single image through model, get probability for each class
    pred_probs = model.predict(img.reshape(1, 96, 96, 1), verbose=0)[0]

    # Class with highest probability is the prediction
    pred_label = int(np.argmax(pred_probs))
    confidence = pred_probs[pred_label] * 100

    # Display image with true vs predicted label
    ax.imshow(img.squeeze(), cmap='gray')
    true_name = class_names[true_label]
    pred_name = class_names[pred_label]

    # Green = correct prediction, Red = wrong prediction
    color = 'green' if pred_label == true_label else 'red'
    ax.set_title(
        "True: {}\nPred: {} ({:.0f}%)".format(true_name, pred_name, confidence),
        fontsize=9, color=color, fontweight='bold'
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig('demo_classification_output.png', dpi=120, bbox_inches='tight')
plt.show()

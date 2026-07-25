import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from ultralytics import YOLO

# ======================== LOAD SAVED MODEL ========================
# Load the trained YOLOv8-nano model (best weights from 20 epochs)
model = YOLO('yolov8_detection/runs/detect/weights/best.pt')

# ======================== LOAD VALIDATION IMAGES ==================
# Validation set — 374 full sonar scenes the model was NOT trained on
# Each image is a 320x480 grayscale sonar scene with multiple objects
val_img_dir = 'yolov8_detection/dataset/val/images'
all_images = sorted(os.listdir(val_img_dir))

# Pick 8 random images (different each run)
np.random.seed(None)
sample_images = np.random.choice(all_images, min(8, len(all_images)), replace=False)

# 11 detection classes and their display colors
CLASS_NAMES = ['Bottle', 'Can', 'Chain', 'Drink-carton', 'Hook',
               'Propeller', 'Shampoo-bottle', 'Standing-bottle',
               'Tire', 'Valve', 'Wall']

COLORS = ['lime', 'red', 'magenta', 'pink', 'white',
          'deepskyblue', 'gold', 'violet', 'cyan', 'yellow', 'orange']

# ======================== DETECT & VISUALIZE ======================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('YOLOv8-nano Detection — Watertank Sonar Scenes', fontsize=14)

for i, img_name in enumerate(sample_images):
    ax = axes[i // 4, i % 4]
    img_path = os.path.join(val_img_dir, img_name)

    # Run detection: returns bounding boxes + class + confidence for each object
    results = model.predict(img_path, imgsz=320, conf=0.25, verbose=False)
    result = results[0]

    # Display the sonar image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ax.imshow(img, cmap='gray')

    # Draw each detected bounding box
    boxes = result.boxes
    for box in boxes:
        # Extract box coordinates (top-left to bottom-right)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].cpu().numpy())       # class index
        conf = float(box.conf[0].cpu().numpy())       # confidence score 0-1
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        color = COLORS[cls_id % len(COLORS)]

        # Draw rectangle around detected object
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # Label with class name and confidence percentage
        ax.text(
            x1, y1 - 3, "{} {:.0f}%".format(cls_name, conf * 100),
            fontsize=7, color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5)
        )

    ax.set_title("{} ({} objects)".format(img_name[:25], len(boxes)), fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('demo_detection_output.png', dpi=120, bbox_inches='tight')
plt.show()

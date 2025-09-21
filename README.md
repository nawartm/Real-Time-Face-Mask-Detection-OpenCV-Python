# Real-Time Face Mask Detection with Deep Learning

> *Objective: Detect whether a person is wearing a face mask in real time via webcam — by combining face detection and image classification using a neural network.*

This project implements a **complete face mask detection system**:
1. **Model Training**: Image classification model (using **MobileNetV2**) trained on a dataset of faces *with* and *without* masks.
2. **Real-Time Detection** via webcam: Face localization + prediction of “Mask” / “No Mask” with probability.

Ideal for **health safety applications**, **access control systems**, or **educational AI projects**.

---

## Target Audience

| Audience | What They Will Find |
|----------|----------------------|
| **Students in AI / Computer Vision** | A complete tutorial, from training to real-time detection, with clear code and explanations. |
| **Teachers / Trainers** | An ideal pedagogical resource for teaching transfer learning, data augmentation, and real-time inference. |
| **Developers / Data Scientists** | A clean implementation using TensorFlow/Keras, OpenCV, and imutils — easy to adapt, deploy, or improve. |
| **Curious Non-Technical Users** | An impressive and practical demo: see how AI can “see” and “understand” what you’re wearing on your face! |

---

## Key Features

### 1. Model Training (`train_mask_detector.py`)
- **Dataset**: Labeled face images (“with_mask” / “without_mask”)
- **Base Model**: **MobileNetV2** (pre-trained on ImageNet) → lightweight and fast
- **Fine-tuning**: Custom classification head added:
  - `AveragePooling2D`
  - `Flatten`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(2, softmax)`
- **Data Augmentation**: Rotations, zooms, shifts, flips → improves generalization
- **Optimization**: Adam optimizer, `lr=1e-4`, learning rate decay
- **Metrics**: Accuracy, Loss (training + validation)
- **Output**: Saved model → `mask_detector.model`

### 2. Real-Time Detection (`detect_mask_video.py`)
- **Face Detection**: **Caffe SSD model** (`res10_300x300_ssd_iter_140000.caffemodel`)
- **Mask Prediction**: Load trained model (`mask_detector.model`)
- **Real-Time Pipeline**:
  1. Video capture (webcam)
  2. Detect all faces in frame
  3. For each face:
     - Extract and resize to 224x224
     - Preprocess (`preprocess_input`)
     - Predict “Mask” / “No Mask” + confidence probability
     - Display label and colored bounding box (green = mask, red = no mask)
- **Control**: Press **‘q’** to quit

---

## Typical Results

- **Validation Accuracy**: > 95% (depending on dataset quality)
- **Detection Speed**: 15–30 FPS on a modern computer (CPU)
- **Training Curves** generated (`plot.png`) → monitor convergence

---

## Technologies & Libraries

```python
# Training
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Detection
from tensorflow.keras.models import load_model
import cv2
from imutils.video import VideoStream
import numpy as np

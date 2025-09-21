# 😷 Détection de Masque Facial en Temps Réel avec Deep Learning

> 🎯 *Objectif : Détecter si une personne porte ou non un masque facial, en direct, via la webcam — en combinant détection de visage et classification d’image avec un réseau de neurones.*

Ce projet implémente un **système complet de détection de masque** :
1. 🧠 **Entraînement d’un modèle** de classification d’images (avec **MobileNetV2**) sur un dataset de visages *avec* et *sans* masque.
2. 🎥 **Détection en temps réel** via webcam : localisation du visage + prédiction “Mask” / “No Mask” avec probabilité.

Parfait pour des applications de **sécurité sanitaire**, **contrôle d’accès**, ou **projets éducatifs en IA**.

---

## 👥 Pour qui est ce projet ?

| Public | Ce qu’il y trouvera |
|--------|----------------------|
| 👩‍🎓 **Étudiants en IA / Vision par ordinateur** | Un tutoriel complet, de l’entraînement à la détection en direct, avec code clair et explications. |
| 👨‍🏫 **Enseignants / Formateurs** | Un support pédagogique idéal pour enseigner le transfert learning, la data augmentation, et l’inférence en temps réel. |
| 👩‍💻 **Développeurs / Data Scientists** | Une implémentation propre avec TensorFlow/Keras, OpenCV, imutils — facile à adapter, déployer ou améliorer. |
| 👔 **Curieux / Non-techniciens** | Une démo impressionnante et utile : voyez comment l’IA peut “voir” et “comprendre” ce que vous portez sur le visage ! |

---

## ⚙️ Fonctionnalités Clés

### 🧪 1. Entraînement du Modèle (`train_mask_detector.py`)
- **Dataset** : Images de visages étiquetées “with_mask” / “without_mask”
- **Modèle de base** : **MobileNetV2** (pré-entraîné sur ImageNet) → léger et rapide
- **Fine-tuning** : Ajout d’une tête de classification personnalisée :
  - `AveragePooling2D`
  - `Flatten`
  - `Dense(128, relu)`
  - `Dropout(0.5)`
  - `Dense(2, softmax)`
- **Data Augmentation** : rotations, zooms, décalages, flips → améliore la généralisation
- **Optimisation** : Adam, `lr=1e-4`, décroissance du taux d’apprentissage
- **Métriques** : Accuracy, Loss (train + validation)
- **Sortie** : Modèle sauvegardé → `mask_detector.model`

### 🎥 2. Détection en Temps Réel (`detect_mask_video.py`)
- **Détection de visage** : Modèle **Caffe SSD** (`res10_300x300_ssd_iter_140000.caffemodel`)
- **Prédiction de masque** : Chargement du modèle entraîné (`mask_detector.model`)
- **Pipeline en temps réel** :
  1. Capture vidéo (webcam)
  2. Détection de tous les visages dans l’image
  3. Pour chaque visage :
     - Extraction + redimensionnement (224x224)
     - Prétraitement (`preprocess_input`)
     - Prédiction “Mask” / “No Mask” + probabilité
     - Affichage du label et du cadre coloré (vert = masque, rouge = pas de masque)
- **Contrôle** : Appuyez sur **‘q’** pour quitter

---

## 📊 Résultats Typiques

- ✅ **Précision de validation** : > 95% (selon la qualité du dataset)
- ⚡ **Vitesse de détection** : 15-30 FPS sur un ordinateur moderne (CPU)
- 📈 **Courbes d’entraînement** générées (`plot.png`) → suivi de la convergence

---

## 🧩 Technologies & Bibliothèques

```python
# Entraînement
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Détection
from tensorflow.keras.models import load_model
import cv2
from imutils.video import VideoStream
import numpy as np

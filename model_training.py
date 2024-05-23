import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# Načtení a příprava dat
data_dir = 'data/raw'
images = []
labels = []  # Toto bude potřeba manuálně označit nebo použít nástroj pro anotaci

for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    image = cv2.imread(img_path)
    image = cv2.resize(image, (128, 128))
    images.append(image)
    # Předpokládaná funkce pro načtení labelů
    labels.append(get_label_from_filename(file))

images = np.array(images)
labels = np.array(labels)

# Definice modelu
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Trénování modelu
model.fit(images, labels, epochs=10, validation_split=0.2)

# Uložení modelu
model.save('src/vision/poker_model.h5')

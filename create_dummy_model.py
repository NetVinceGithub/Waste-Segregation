import tensorflow as tf
from tensorflow import keras
import numpy as np

print("Creating a dummy model for testing...")

# Create a simple model
model = keras.Sequential([
    keras.layers.Input(shape=(224, 224, 3)),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(3, activation='softmax')  # 3 classes: BIO, NON-BIO, RECYCLABLE
])

# Compile it
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save it
model.save('garbage_model.h5')
print("✅ Dummy model created: garbage_model.h5")
print("⚠️ Note: This is a dummy model - predictions will be random!")
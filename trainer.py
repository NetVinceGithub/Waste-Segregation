import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50  
LEARNING_RATE = 0.001

if not os.path.exists("dataset"):
    print("Error: 'dataset' folder not found!")
    print("Create folder structure:")
    print("dataset/")
    print("  ├── bio/")
    print("  ├── nonbio/")
    print("  ├── recyclable/")
    exit()

for folder in ["bio", "nonbio", "recyclable"]:
    path = f"dataset/{folder}"
    if not os.path.exists(path):
        print(f"Error: Missing folder 'dataset/{folder}'")
        exit()
    
    num_images = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{folder}: {num_images} images")
    
    if num_images < 50:
        print(f"Warning: {folder} has only {num_images} images. Recommended: 100+")

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")


base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)

for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

print(f"Base model loaded (trainable layers: {sum([1 for l in base_model.layers if l.trainable])})")

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)  
x = Dropout(0.5)(x)  
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(3, activation="softmax")(x)  # Changed from 2 to 3 classes

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled")


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,          
    width_shift_range=0.2,     
    height_shift_range=0.2,     
    shear_range=0.2,            
    zoom_range=0.2,            
    horizontal_flip=True,      
    fill_mode="nearest",
    validation_split=0.2
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("Data augmentation configured")

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    "dataset",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Classes: {train_data.class_indices}")


early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

print("Callbacks configured")


print("\n" + "="*50)
print("Training started... This may take 10-30 minutes")
print("="*50 + "\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("\n" + "="*50)
print("Training completed!")
print("="*50)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"\nFinal Training Accuracy: {final_train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc*100:.2f}%")

if final_val_acc < 0.7:
    print("\nWarning: Validation accuracy is below 70%")
    print("Recommendations:")
    print("  - Collect more images (aim for 200+ per class)")
    print("  - Ensure images are clear and well-lit")
    print("  - Use diverse items in each category")
elif final_val_acc < 0.85:
    print("\nModel is decent but could be improved")
    print("Consider collecting more diverse training data")
else:
    print("\nExcellent accuracy! Model should work well")


model.save("garbage_model.h5")
print("\nModel saved as 'garbage_model.h5'")
print("\nYou can now run your detection script!")
print("="*50)
import tensorflow as tf
import os

# Paths to the dataset
dataset_dir = "dataset"  # Adjust if the dataset is in a different directory
train_dir = os.path.join(dataset_dir, "Train")
test_dir = os.path.join(dataset_dir, "Test")

# Data Preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                                shear_range=0.2, 
                                                                zoom_range=0.2, 
                                                                horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(train_dir, target_size=(64, 64), batch_size=32, class_mode="binary")
test_data = test_datagen.flow_from_directory(test_dir, target_size=(64, 64), batch_size=32, class_mode="binary")

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_data, epochs=5, validation_data=test_data)

# Save Model
os.makedirs("D:/Project/backend/saved_model", exist_ok=True)
model.save("D:/Project/backend/saved_model.keras")





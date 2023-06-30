# importing libraries
from pathlib import Path

import matplotlib.pyplot as plt  # Matplotlib module for data visualization
import tensorflow as tf  # TensorFlow module for deep learning
from keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam  # Optimizer for model training
from tensorflow.keras.preprocessing.image import ImageDataGenerator

flowers = Path.cwd() / "Downloads" / "flowers" / "flowers"

images = []
labels = []

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

train_ds = datagen.flow_from_directory(
    flowers,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    seed=123,
)
val_ds = datagen.flow_from_directory(
    flowers,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    seed=123,
)


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(150, 150, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()


# Train the model
batch_size = 32
epochs = 100

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

history = (
    model.fit(  # train the model for a fixed number of epochs (iterations on a dataset)
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[early_stopping],
    )
)

# save the model
model.save("progetto_silvia.h5")

# plot the training and validation accuracy and loss at each epoch
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")
plt.show()


test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print("\nTest loss=", test_loss)
print("\nTest accuracy=", test_acc)

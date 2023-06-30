import tensorflow as tf
from keras.applications.efficientnet_v2 import EfficientNetV2B0
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

# Set the number of folds for k-fold cross validation
num_folds = 5

# Set the batch size and number of epochs for training
batch_size = 32
epochs = 10

# Load the dataset
flowers = r"C:\Users\rober\Downloads\flowers\flowers"
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)
val_datagen = ImageDataGenerator(validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    flowers,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    seed=42,
)
validation_generator = val_datagen.flow_from_directory(
    flowers,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    seed=42,
)


# Create a KFold object with the desired number of folds
kf = KFold(n_splits=num_folds, shuffle=True)

# Loop over the folds
for fold, (train_indices, val_indices) in enumerate(kf.split(train_generator)):
    print(f"Fold {fold+1}")

    # Create a new model for each fold
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
        optimizer=Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Create a new training data generator for each fold
    train_data = train_datagen.flow_from_directory(
        flowers,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=False,
        seed=fold,
    )
    val_data = train_datagen.flow_from_directory(
        flowers,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=fold,
    )

    # Train the model for the specified number of epochs
    model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
    )

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(validation_generator)
print(f"Validation loss: {loss}")
print(f"Validation accuracy: {accuracy}")

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


np.random.seed(1)
sns.set(style="white", context="notebook", palette="pastel")

# Load
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Preprocess
x_train = train.drop(labels=["label"], axis=1)
x_train /= 255.0
x_train = x_train.values.reshape(-1, 28, 28, 1)

y_train = train["label"]
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

test /= 255.0
test = test.values.reshape(-1, 28, 28, 1)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, random_state=1
)

# Build model
model = tf.keras.Sequential()
model.add(
    layers.Conv2D(filters=32, kernel_size=3, activation="relu", input_shape=(28, 28, 1))
)
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=32, kernel_size=3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=32, kernel_size=5, strides=2, padding="same", activation="relu"
    )
)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(filters=64, kernel_size=3, activation="relu"))
model.add(layers.BatchNormalization())
model.add(
    layers.Conv2D(
        filters=64, kernel_size=5, strides=2, padding="same", activation="relu"
    )
)
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(10, activation="softmax"))

model.summary()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Generate augmented data
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
)

# Create callback for model
annealer = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

# Run the model
batch_size = 60
steps_per_epoch = x_train.shape[0] // batch_size
history = model.fit_generator(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    verbose=1,
    callbacks=[annealer],
    validation_data=(x_val, y_val),
)

# Save weights
model.save_weights("weights.h5")

# Evaluate model
loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
print("Loss: {0:.6f}, Accuracy: {1:.6f}".format(loss, accuracy))

# Plot accuracy and loss
epochs = range(1, len(history.history["accuracy"]) + 1)
plt.plot(epochs, history.history["accuracy"], "ro", label="training acc")
plt.plot(epochs, history.history["val_accuracy"], "b", label="val accuracy")
plt.title("training and val accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, history.history["loss"], "ro", label="training loss")
plt.plot(epochs, history.history["val_loss"], "b", label="val loss")
plt.title("training and val loss")
plt.legend()
plt.figure()

plt.show()

# Show confusion matrix
y_hat = model.predict(x_val)
y_pred = np.argmax(y_hat, axis=1)
y_true = np.argmax(y_val, axis=1)
cm = tf.math.confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

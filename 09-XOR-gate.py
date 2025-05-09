import tensorflow as tf
from tensorflow import keras
import numpy as np

# XOR dataset
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([[0], [1], [1], [0]])

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),  # Fixed learning rate type
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(x, xor_y, epochs=100, batch_size=4)

# Evaluate
loss, accuracy = model.evaluate(x, xor_y)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predict
predictions = model.predict(x)
print(f'X:\n{x}')
print(f'Predictions:\n{predictions}')
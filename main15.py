import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Loading the dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model (Adding more layers and increasing neurons)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Optimizer tuning
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# Adam Optimizer
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=20)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Predict on new data
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)

# Example: print the prediction for the first test image
print("Predicted label:", tf.argmax(predictions[0]).numpy())
print("True label:", y_test[0])

# Visualization
plt.figure(figsize=(4,4))
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Predicted: {tf.argmax(predictions[0]).numpy()}, True: {y_test[0]}")
plt.axis('off')
plt.show()

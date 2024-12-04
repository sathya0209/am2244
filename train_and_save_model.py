import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the saved model
model = tf.keras.models.load_model("mnist_digit_recognizer.h5")
print("Model loaded successfully!")

# Load the MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0

# Test the model on a random sample
random_index = np.random.randint(0, x_test.shape[0])
sample_image = x_test[random_index]
sample_label = y_test[random_index]

prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_digit = np.argmax(prediction)

print(f"True Label: {sample_label}")
print(f"Predicted Digit: {predicted_digit}")

# Visualize the sample
plt.imshow(sample_image, cmap="gray")
plt.title(f"True: {sample_label}, Predicted: {predicted_digit}")
plt.show()

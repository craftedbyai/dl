# Experiment 6

## Aim

To design the architecture and implement a **Convolutional Neural Network (CNN)** model for handwritten **digit recognition using the MNIST dataset**.

---

# Theory (Improved Version)

A **Convolutional Neural Network (CNN)** is a specialized type of deep learning model primarily used for **image processing and computer vision tasks** such as image classification, object detection, and pattern recognition.

Unlike traditional neural networks, CNNs are designed to automatically and adaptively learn **spatial hierarchies of features** from images. This makes them extremely effective for analyzing visual data.

A typical CNN architecture consists of several types of layers:

### 1. Convolutional Layer

The convolutional layer is the core building block of a CNN. It applies multiple **filters (kernels)** to the input image to extract important features such as edges, textures, and patterns. Each filter produces a **feature map** that highlights specific characteristics of the image.

Mathematically, convolution is performed by sliding the filter across the input image and computing the dot product between the filter and the input pixels.

### 2. Activation Function

After convolution, an activation function such as **ReLU (Rectified Linear Unit)** is applied. ReLU introduces non-linearity into the model and helps the network learn complex patterns.

ReLU function:
[
f(x) = max(0,x)
]

### 3. Pooling Layer

Pooling layers reduce the spatial dimensions of the feature maps. This helps to:

* Reduce computational complexity
* Prevent overfitting
* Retain important features

The most common pooling operation is **Max Pooling**, which selects the maximum value from a region of the feature map.

### 4. Flatten Layer

The flatten layer converts the 2D feature maps into a **1D feature vector**, allowing it to be passed into fully connected layers.

### 5. Fully Connected Layer

The fully connected (dense) layer performs classification using the extracted features. It connects every neuron in one layer to every neuron in the next layer.

### 6. Softmax Output Layer

The final layer uses **Softmax activation**, which converts outputs into **probabilities for each class**.

In digit recognition, the network predicts one of **10 digits (0–9)**.

### MNIST Dataset

The **MNIST dataset** is one of the most widely used datasets for image classification. It contains:

* 60,000 training images
* 10,000 testing images
* Image size: **28 × 28 pixels**
* 10 classes representing digits **0–9**

CNN models trained on MNIST typically achieve very high accuracy because handwritten digits contain distinct spatial patterns that CNNs can effectively learn.

---

# Python Code (Run in Any IDE)

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshape data to include channel dimension
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize pixel values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("Training data shape:", x_train.shape)
print("Testing data shape:", x_test.shape)

# Build CNN Model
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Show architecture
model.summary()

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=10,
    validation_split=0.2
)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_acc)
print("Test Loss:", test_loss)

# Plot training history
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# Show predictions on test images
plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4,4,i+1)

    img = x_test[i].reshape(1,28,28,1)
    pred = model.predict(img, verbose=0)

    predicted_label = pred.argmax()
    true_label = np.argmax(y_test[i])

    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"Pred: {predicted_label} True: {true_label}")
    plt.axis('off')

plt.show()
```

---

# Output

* Model summary displaying CNN architecture
* Training accuracy and validation accuracy graphs
* Training loss and validation loss graphs
* Prediction results for sample test images
* Final **test accuracy (usually around 98–99%)**

---

# Conclusion (Better Version)

The Convolutional Neural Network (CNN) model was successfully implemented for handwritten digit recognition using the MNIST dataset. The model effectively learned spatial features of images using convolutional and pooling layers. After training, the model achieved high accuracy in classifying digits from 0 to 9. This experiment demonstrates how CNNs can automatically extract meaningful features from images and perform highly accurate classification tasks.

---

If you want, I can also give you **a much better theory + clean runnable code for Experiment 7 (LSTM Sentiment Analysis)** so your **entire practical file becomes perfect for viva and submission.**

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize images
train_images = train_images / 255.0
test_images = test_images / 255.0

# Class names
class_names = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Create CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
print("Training Model...")
model.fit(train_images, train_labels, epochs=5)

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Predict sample image
prediction = model.predict(test_images[:1])
predicted_class = class_names[prediction.argmax()]

print("Predicted Class:", predicted_class)

# Display image
plt.imshow(test_images[0])
plt.title(f"Prediction: {predicted_class}")
plt.axis("off")
plt.show()

import os
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input,
              outputs=base_model.layers[-2].output)

image_folder = "images"
features = {}

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    feature = model.predict(img, verbose=0)
    features[img_name] = feature

print("Image Feature Extraction Completed!")

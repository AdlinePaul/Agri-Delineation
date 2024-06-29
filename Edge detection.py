import numpy as np
import cv2
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Activation
from keras.optimizers import Adam
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Placeholder functions for loading satellite and SAR images
def load_satellite_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image / 255.0  # Normalize

def load_sar_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None
    return image / 255.0  # Normalize

satellite_image_path = 'satelliteimg.jpg'
sar_image_path = 'sarimg.jpg'

# Load images
satellite_image = load_satellite_image(satellite_image_path)
sar_image = load_sar_image(sar_image_path)

if satellite_image is not None and sar_image is not None:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(satellite_image)
    plt.title('Satellite Image')
    plt.subplot(1, 2, 2)
    plt.imshow(sar_image, cmap='gray')
    plt.title('SAR Image')
    plt.show()
else:
    print("Failed to load one or both images.")

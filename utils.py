import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

MODEL_PATH = 'model/updated__brain__tumor__segmentation.h5'
model = load_model(MODEL_PATH)

def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def postprocess_and_save(pred_mask, filename):
    pred_mask = pred_mask[0, :, :, 0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    save_path = f'static/results/{filename}'
    cv2.imwrite(save_path, pred_mask)
    return save_path

def load_model_and_predict(image_path, filename):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    return postprocess_and_save(prediction, filename)
    img = cv2.imread(image_path)

    # Resize and normalize
    img = cv2.resize(img, (256, 256))
    img = img / 255.0

    # Expand dims to add batch
    img = np.expand_dims(img, axis=0)  # shape: (1, 256, 256, 3)

from PIL import Image
import numpy as np

def preprocess_image(image_path):
    image = Image.open(image_path)

    # Convert to RGB (3 channels) from any mode like L, LA, RGBA etc.
    image = image.convert('RGB')

    # Resize image to match model input
    image = image.resize((256, 256))

    # Convert image to array and normalize
    image_array = np.array(image) / 255.0

    # Add batch dimension: (1, 256, 256, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array



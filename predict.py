# ==============================================================================
# predict.py
#
# Script to predict the disease from a single X-ray image using a trained model.
# Usage: python predict.py --image path/to/your/image.jpeg
# ==============================================================================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import argparse

# --- Configuration ---
IMG_WIDTH, IMG_HEIGHT = 224, 224
MODEL_PATH = "multi_disease_xray_detector.h5"
# IMPORTANT: The order of these labels MUST match the order from the training generator.
# You can find the order printed when you run train_model.py
CLASS_LABELS = ['Effusion', 'Fibrosis', 'Pneumonia']  # Example: Update with your actual labels


def predict_disease_from_xray(image_path, model_path, labels):
    """
    Takes the file path of an X-ray image and returns the predicted disease.
    """
    try:
        # 1. Load the trained AI model
        model = load_model(model_path)

        # 2. Load and preprocess the single image
        image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image)
        image_array = image_array / 255.0
        image_batch = np.expand_dims(image_array, axis=0)

        # 3. Make the prediction
        prediction_probabilities = model.predict(image_batch)[0]

        # 4. Interpret the result
        predicted_index = np.argmax(prediction_probabilities)
        predicted_disease = labels[predicted_index]
        confidence = prediction_probabilities[predicted_index]

        print("\n--- Analysis Result ---")
        print(f"Predicted Disease: {predicted_disease}")
        print(f"Confidence: {confidence:.2%}")

        return predicted_disease, confidence

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have trained the model by running 'train_model.py' first.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Set up an argument parser to accept an image path from the command line
    parser = argparse.ArgumentParser(description="Predict disease from a chest X-ray image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the X-ray image file.")

    args = parser.parse_args()

    # Run the prediction function
    predict_disease_from_xray(args.image, MODEL_PATH, CLASS_LABELS)

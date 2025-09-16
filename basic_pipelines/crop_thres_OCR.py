import cv2
import os
import numpy as np
from datetime import datetime
import pytesseract
from difflib import SequenceMatcher

# Path to the input image
IMAGE_PATH = "/home/scalepi/Desktop/savephototest/chip_20241205_141703.png"

# Normalized bounding box coordinates (0 to 1)
normalized_bbox = (0.73, 0.44, 0.90, 0.74)  # Example: xmin, ymin, xmax, ymax

# Folder to save the preprocessed image (optional for debugging)
SAVE_FOLDER = "/home/scalepi/Desktop/OCR"

# List of known part numbers
KNOWN_PART_NUMBERS = [
    "SN74LS03", "SN74LS51NB180525", "SM74KS59MB18034", "SN74LSS1NB1B035"
]

def crop_and_threshold(image_path, bbox, save_folder=None):
    """
    Crop the image and apply dynamic thresholding for OCR preparation.

    Parameters:
        image_path (str): Path to the input image.
        bbox (tuple): Normalized bounding box (xmin, ymin, xmax, ymax).
        save_folder (str): Path to the folder where preprocessed image will be saved (optional).

    Returns:
        numpy.ndarray: Preprocessed image ready for OCR.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to read image at {image_path}")

        # Image dimensions
        height, width, _ = image.shape

        # Convert normalized coordinates to pixel values
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Convert to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Compute histogram and calculate dynamic threshold
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        cumulative_sum = np.cumsum(hist).flatten()

        # Find the intensity where cumulative sum is approximately 90% of total pixels
        threshold = np.argmax(cumulative_sum >= 0.42 * total_pixels)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Save the thresholded image for debugging (optional)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            debug_filename = os.path.join(save_folder, f"prepped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(debug_filename, thresholded_image)
            print(f"Preprocessed image saved at {debug_filename}")

        return thresholded_image

    except Exception as e:
        raise RuntimeError(f"Failed to crop and preprocess image: {e}")

def detect_text_in_image(image):
    """
    Detect text in the preprocessed image using Tesseract OCR.

    Parameters:
        image (numpy.ndarray): Preprocessed image ready for OCR.

    Returns:
        str: Detected text from the image.
    """
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6'
    detected_text = pytesseract.image_to_string(image, config=custom_config)
    return detected_text.strip()

def calculate_similarity(a, b):
    """
    Calculate similarity between two strings.

    Parameters:
        a (str): First string.
        b (str): Second string.

    Returns:
        float: Similarity ratio (0 to 1).
    """
    return SequenceMatcher(None, a, b).ratio()

def find_closest_part(detected_text, part_numbers):
    """
    Find the closest matching part number from the list.

    Parameters:
        detected_text (str): Text detected from the image.
        part_numbers (list): List of known part numbers.

    Returns:
        tuple: Closest part number and its similarity percentage.
    """
    closest_part = None
    highest_similarity = 0

    for part in part_numbers:
        similarity = calculate_similarity(detected_text, part)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_part = part

    return closest_part, highest_similarity * 100

# Main function
if __name__ == "__main__":
    try:
        # Step 1: Crop and preprocess the image
        preprocessed_image = crop_and_threshold(IMAGE_PATH, normalized_bbox, SAVE_FOLDER)

        # Step 2: Detect text in the preprocessed image
        detected_text = detect_text_in_image(preprocessed_image)
        print("Detected Text:\n", detected_text)

        # Step 3: Find the closest part number
        closest_part, similarity_percentage = find_closest_part(detected_text, KNOWN_PART_NUMBERS)

        print(f"Closest Part Number: {closest_part}")
        print(f"Similarity: {similarity_percentage:.2f}%")

    except Exception as e:
        print(e)

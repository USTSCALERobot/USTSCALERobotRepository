import cv2
import os
import numpy as np
from datetime import datetime

# Path to the input image
IMAGE_PATH = "/home/scalepi/Desktop/savephototest/chip_20241205_141703.png"
SAVE_FOLDER = "/home/scalepi/Desktop/prepped"

# Normalized bounding box coordinates (0 to 1)
normalized_bbox = (0.73, 0.44, 0.90, 0.74)  # Example: xmin, ymin, xmax, ymax

def crop_and_threshold(image_path, bbox, save_folder):
    """
    Crop the image and apply dynamic thresholding for OCR preparation.

    Parameters:
        image_path (str): Path to the input image.
        bbox (tuple): Normalized bounding box (xmin, ymin, xmax, ymax).
        save_folder (str): Path to the folder where preprocessed image will be saved.

    Returns:
        str: Path to the saved preprocessed image.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return None

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
        threshold = np.argmax(cumulative_sum >= 0.4 * total_pixels)

        # Apply thresholding
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Save the thresholded image with a timestamped filename
        filename = os.path.join(save_folder, f"prepped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, thresholded_image)

        print(f"Preprocessed image saved at {filename}")
        return filename

    except Exception as e:
        print(f"Failed to crop and preprocess image: {e}")
        return None

# Run the cropping and preprocessing function
if __name__ == "__main__":
    crop_and_threshold(IMAGE_PATH, normalized_bbox, SAVE_FOLDER)

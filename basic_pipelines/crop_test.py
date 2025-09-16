import cv2
import os
from datetime import datetime

# Path to the input image
IMAGE_PATH = "/home/scalepi/Desktop/savephototest/chip_20241205_134041.png"
SAVE_FOLDER = "/home/scalepi/Desktop/cropped"

# Normalized bounding box coordinates (0 to 1)
normalized_bbox = (0.82, 0.41, 0.92, 0.58)  # Example: xmin, ymin, xmax, ymax

def crop_and_save_image(image_path, bbox, save_folder):
    """
    Crop and save the region of the image specified by the bounding box.

    Parameters:
        image_path (str): Path to the input image.
        bbox (tuple): Normalized bounding box (xmin, ymin, xmax, ymax).
        save_folder (str): Path to the folder where cropped image will be saved.

    Returns:
        str: Path to the saved cropped image.
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

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Save the cropped image with a timestamped filename
        filename = os.path.join(save_folder, f"cropped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, cropped_image)

        print(f"Cropped image saved at {filename}")
        return filename

    except Exception as e:
        print(f"Failed to crop and save image: {e}")
        return None

# Run the cropping function
if __name__ == "__main__":
    crop_and_save_image(IMAGE_PATH, normalized_bbox, SAVE_FOLDER)
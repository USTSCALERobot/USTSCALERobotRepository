import os
import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher
from datetime import datetime
import subprocess
import time

# Paths
CHIP_DETECTION_DIR = "/home/scalepi/hailo-rpi5-examples"
HAILO_ENV_PATH = CHIP_DETECTION_DIR + "/setup_env.sh"
OCR_VENV_PATH = "~/myenv/bin/activate"
SAVE_FOLDER = "/home/scalepi/Desktop/New"
OCR_SAVE_FOLDER = "/home/scalepi/Desktop/New"
IMAGE_PATH = os.path.join(SAVE_FOLDER, "chip_detected.png")

# Known Part Numbers
KNOWN_PART_NUMBERS = [
    "SN74LS03", "SN74LS51NB180525", "SM74KS59MB18034", "SN74LSS1NB1B035"
]

# Global variable for the last bounding box
last_bbox = None


def run_chip_detection():
    """Runs the AI chip detection model inside the Hailo environment."""
    print("🔍 Running AI Chip Detection...")

    try:
        command = f"cd {CHIP_DETECTION_DIR} && source {HAILO_ENV_PATH} && python3 basic_pipelines/lucas_test.py"
        subprocess.run(command, shell=True, executable="/bin/bash")
    except Exception as e:
        print(f"❌ Failed to run AI detection: {e}")


def crop_and_threshold(image_path, bbox, save_folder=None):
    """Crops and preprocesses the detected chip area for OCR."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to read image at {image_path}")

        height, width, _ = image.shape
        x1, y1, x2, y2 = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        cropped_image = image[y1:y2, x1:x2]

        # Convert to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        # Compute histogram-based threshold
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        threshold = np.argmax(np.cumsum(hist).flatten() >= 0.47 * (gray_image.size))
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

        # Save for debugging
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            debug_filename = os.path.join(save_folder, "preprocessed.png")
            cv2.imwrite(debug_filename, thresholded_image)

        return thresholded_image

    except Exception as e:
        raise RuntimeError(f"❌ Failed to preprocess image: {e}")


def detect_text(image):
    """Runs Tesseract OCR on the cropped and preprocessed image."""
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6'
    return pytesseract.image_to_string(image, config=custom_config).strip()


def find_closest_part(detected_text, part_numbers):
    """Finds the closest matching part number from the known list."""
    best_match = max(part_numbers, key=lambda p: SequenceMatcher(None, detected_text, p).ratio(), default=None)
    return best_match, SequenceMatcher(None, detected_text, best_match).ratio() * 100 if best_match else 0


def run_ocr():
    """Runs the OCR process inside the correct virtual environment."""
    print("📄 Running OCR...")

    global last_bbox

    # Wait to ensure detection has finished
    time.sleep(1)

    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Image {IMAGE_PATH} not found. Skipping OCR.")
        return

    if last_bbox is None:
        print("❌ No bounding box available. Skipping OCR.")
        return

    try:
        # Run OCR inside the virtual environment
        command = f"source ~/myenv/bin/activate && python3 /basic_pipelines/crop_thres_OCR_2.py"
        subprocess.run(command, shell=True, executable="/bin/bash")
    except Exception as e:
        print(f"❌ Failed to run OCR: {e}")

if __name__ == "__main__":
    # Step 1: Run AI Chip Detection
    run_chip_detection()

    # Step 2: Activate OCR Environment & Run OCR
    try:
        command = f"source {OCR_VENV_PATH} && python3 /home/scalepi/hailo-rpi5-examples/basic_pipelines/crop_thres_OCR_2.py"
        subprocess.run(command, shell=True, executable="/bin/bash")
    except Exception as e:
        print(f"❌ Failed to start OCR: {e}")
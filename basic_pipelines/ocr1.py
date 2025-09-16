#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime
import cv2
import numpy as np
import pytesseract
from difflib import SequenceMatcher
import concurrent.futures
import os

# --- Parse Command-line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="OCR Processing Script")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--bbox", type=str, required=True,
                        help="Normalized bounding box as comma-separated values: xmin,ymin,xmax,ymax")
    parser.add_argument("--save_folder", type=str, default="/home/scalepi/Desktop/OCR",
                        help="Folder to save preprocessed images")
    return parser.parse_args()

args = parse_args()
IMAGE_PATH = args.image
try:
    normalized_bbox = tuple(map(float, args.bbox.split(",")))
    if len(normalized_bbox) != 4:
        raise ValueError
except Exception:
    sys.exit("Error: --bbox must have 4 comma-separated float values.")
SAVE_FOLDER = args.save_folder

# --- OCR Functions ---
def crop_and_threshold(image_path, bbox, save_folder=None):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to read image at {image_path}")
        height, width, _ = image.shape
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        cropped_image = image[y1:y2, x1:x2]
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0,256])
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        cumulative_sum = np.cumsum(hist).flatten()
        threshold = int(np.argmax(cumulative_sum >= 0.35 * total_pixels))
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            debug_filename = os.path.join(save_folder, f"prepped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(debug_filename, thresholded_image)
            print(f"Preprocessed image saved at {debug_filename}")
        return thresholded_image
    except Exception as e:
        raise RuntimeError(f"Failed to crop and preprocess image: {e}")

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

def rotate_to_align_long_side(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    if height > width:
        image = rotate_image(image, -90)
    return image

def detect_text_in_image(image):
    custom_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 --psm 6'
    detected_text = pytesseract.image_to_string(image, config=custom_config)
    return detected_text.strip()

def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_closest_part(detected_text, part_numbers):
    closest_part = None
    highest_similarity = 0
    for part in part_numbers:
        similarity = calculate_similarity(detected_text, part)
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_part = part
    return closest_part, highest_similarity * 100

def crop_and_threshold_with_rotation(image_path, bbox, save_folder=None):
    try:
        preprocessed_image = crop_and_threshold(image_path, bbox, save_folder)
        aligned_image = rotate_to_align_long_side(preprocessed_image, bbox)
        rotated_180 = rotate_image(aligned_image, 180)
        if save_folder:
            aligned_filename = os.path.join(save_folder, f"aligned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            rotated_filename = os.path.join(save_folder, f"rotated_180_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(aligned_filename, aligned_image)
            cv2.imwrite(rotated_filename, rotated_180)
            print(f"Saved aligned image at {aligned_filename}")
            print(f"Saved rotated image at {rotated_filename}")
        return [aligned_image, rotated_180]
    except Exception as e:
        raise RuntimeError(f"Failed to crop and preprocess image with rotation: {e}")

def run_ocr():
    try:
        preprocessed_images = crop_and_threshold_with_rotation(IMAGE_PATH, normalized_bbox, SAVE_FOLDER)
        best_match = None
        highest_similarity = 0
        for idx, image in enumerate(preprocessed_images):
            orientation = "aligned with long side at bottom" if idx == 0 else "rotated 180 degrees"
            detected_text = detect_text_in_image(image)
            closest_part, similarity_percentage = find_closest_part(detected_text, [
                "SN74LS03", "SN74LS51NB180525", "SM74KS59MB18034", "SN74LSS1NB1B035"
            ])
            print(f"Orientation: {orientation}")
            print(f"Detected Text: {detected_text}")
            print(f"Closest Part Number: {closest_part}")
            print(f"Similarity: {similarity_percentage:.2f}%\n")
            if similarity_percentage > highest_similarity:
                highest_similarity = similarity_percentage
                best_match = (detected_text, closest_part, similarity_percentage, orientation)
        if best_match:
            detected_text, closest_part, similarity_percentage, orientation = best_match
            print("Best Match:")
            print(f"Orientation: {orientation}")
            print(f"Detected Text: {detected_text}")
            print(f"Closest Part Number: {closest_part}")
            print(f"Similarity: {similarity_percentage:.2f}%")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    run_ocr()
#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import gc

# =============================================================================
# Step 0. Ensure we are running in the desired virtual environment.
# =============================================================================
DESIRED_VENV_PYTHON = "/home/scalepi/hailo-rpi5-examples/venv_hailo_rpi5_examples/bin/python3"
if not sys.executable.startswith(DESIRED_VENV_PYTHON):
    print("Re-executing using the virtual environment's python...")
    os.execv(DESIRED_VENV_PYTHON, [DESIRED_VENV_PYTHON] + sys.argv)

# =============================================================================
# Step 1. Auto-Activate Environment Variables
# =============================================================================
def activate_env():
    if os.getenv("HAILO_ENV_ACTIVATED") == "1":
        print("✅ Environment already activated.")
        return
    print("🔧 Activating environment...")
    SETUP_ENV_SCRIPT = "/home/scalepi/hailo-rpi5-examples/setup_env.sh"
    VENV_ACTIVATE = "/home/scalepi/hailo-rpi5-examples/venv_hailo_rpi5_examples/bin/activate"
    command = (
        f"bash -c 'source {SETUP_ENV_SCRIPT} && "
        f"source {VENV_ACTIVATE} && "
        f"export HAILO_ENV_ACTIVATED=1 && env'"
    )
    result = subprocess.run(command, shell=True, executable="/bin/bash", capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ Error activating environment:", result.stderr)
        sys.exit(1)
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value
    if "TAPPAS_POST_PROC_DIR" not in os.environ:
        os.environ["TAPPAS_POST_PROC_DIR"] = "/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes"
    print("✅ Environment activated successfully.")

activate_env()

# =============================================================================
# Step 2. Imports for Both AI and OCR
# =============================================================================
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import cv2
import numpy as np
from datetime import datetime
import hailo
from hailo_rpi_common import get_caps_from_pad, app_callback_class
from detection_pipeline import GStreamerDetectionApp

try:
    import pytesseract
    from difflib import SequenceMatcher
except ModuleNotFoundError as e:
    print(f"Module error: {e}\nPlease install the missing module(s) (e.g. pip install pytesseract) in your virtual environment.")
    sys.exit(1)

# =============================================================================
# Step 3. Global Paths and Constants
# =============================================================================
SAVE_FOLDER = "/home/scalepi/Desktop/savephototest"
AI_IMAGE_PATH = os.path.join(SAVE_FOLDER, "chip_detected.png")
OCR_IMAGE_PATH = AI_IMAGE_PATH
normalized_bbox = (0.65, 0.33, 0.90, 0.47)
KNOWN_PART_NUMBERS = [
    "SN74LS03", "SN74LS51NB180525", "SM74KS59MB18034", "SN74LSS1NB1B035"
]

# =============================================================================
# Step 4. AI Detection Code (Using GStreamerDetectionApp)
# =============================================================================
class UserAppCallback(app_callback_class):
    def __init__(self, pipeline, main_loop):
        super().__init__()
        self.stop_detection = False
        self.pipeline = pipeline
        self.main_loop = main_loop

    def stop_and_save(self, frame):
        save_frame_ai(frame)
        stop_pipeline(self.pipeline, self.main_loop)

def save_frame_ai(frame):
    try:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        cv2.imwrite(AI_IMAGE_PATH, frame_bgr)
        print(f"📸 AI: Frame saved at {AI_IMAGE_PATH}")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ AI: Failed to save frame: {e}")

def extract_raw_frame(buffer, width, height):
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("❌ AI: Failed to map buffer.")
        return None
    try:
        raw_data = np.frombuffer(map_info.data, dtype=np.uint8)
        return raw_data.reshape((height, width, 3))
    except Exception as e:
        print(f"❌ AI: Error extracting raw frame: {e}")
        return None
    finally:
        buffer.unmap(map_info)

def app_callback(pad, info, user_data):
    if user_data.stop_detection:
        return Gst.PadProbeReturn.DROP
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK
    format, width, height = get_caps_from_pad(pad)
    if not format or not width or not height:
        return Gst.PadProbeReturn.OK
    frame = extract_raw_frame(buffer, width, height)
    if frame is None:
        return Gst.PadProbeReturn.OK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        x1, y1, x2, y2 = bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()
        print(f"🔍 AI: Detection - Label: {label}, BBox: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}), Confidence: {confidence:.2f}")
        if x2 >= 0.9:
            print(f"✅ AI: Condition met: x_max = {x2:.2f} >= 0.9")
            user_data.stop_and_save(frame)
            return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK

def stop_pipeline(pipeline, main_loop):
    try:
        print("🛑 AI: Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        Gst.deinit()
        print("✅ AI: Pipeline stopped.")
        if main_loop:
            main_loop.quit()
    except Exception as e:
        print(f"❌ AI: Failed to stop pipeline: {e}")

def run_ai_detection():
    if os.getenv("HAILO_ENV_ACTIVATED") != "1":
        print("❌ AI: Hailo environment is NOT activated!")
        sys.exit(1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ AI: Camera not detected!")
        sys.exit(1)
    cap.release()
    print("✅ AI: Camera detected successfully. Starting AI pipeline...")
    os.chdir("/home/scalepi/hailo-rpi5-examples")
    Gst.init(None)
    main_loop = GLib.MainLoop()
    dummy_pipeline = Gst.Pipeline.new("dummy-pipeline")
    user_data = UserAppCallback(dummy_pipeline, main_loop)
    app = GStreamerDetectionApp(app_callback, user_data)
    user_data.pipeline = app.pipeline
    print("🚀 AI: Running detection pipeline...")
    app.run()
    main_loop.run()
    print("✅ AI: Detection finished.")

# =============================================================================
# Step 5. OCR Processing Code
# =============================================================================
def crop_and_threshold(image_path, bbox, save_folder=None):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"OCR: Unable to read image at {image_path}")
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
        threshold = np.argmax(cumulative_sum >= 0.47 * total_pixels)
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            debug_filename = os.path.join(save_folder, f"prepped_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            cv2.imwrite(debug_filename, thresholded_image)
            print(f"OCR: Preprocessed image saved at {debug_filename}")
        return thresholded_image
    except Exception as e:
        raise RuntimeError(f"OCR: Failed to crop and preprocess image: {e}")

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
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
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
            print(f"OCR: Saved aligned image at {aligned_filename}")
            print(f"OCR: Saved rotated image at {rotated_filename}")
        return [aligned_image, rotated_180]
    except Exception as e:
        raise RuntimeError(f"OCR: Failed to crop and preprocess image with rotation: {e}")

def run_ocr():
    try:
        preprocessed_images = crop_and_threshold_with_rotation(OCR_IMAGE_PATH, normalized_bbox, SAVE_FOLDER)
        best_match = None
        highest_similarity = 0
        for idx, image in enumerate(preprocessed_images):
            orientation = "aligned with long side at bottom" if idx == 0 else "rotated 180 degrees"
            detected_text = detect_text_in_image(image)
            closest_part, similarity_percentage = find_closest_part(detected_text, KNOWN_PART_NUMBERS)
            print(f"OCR: Orientation: {orientation}")
            print(f"OCR: Detected Text: {detected_text}")
            print(f"OCR: Closest Part Number: {closest_part}")
            print(f"OCR: Similarity: {similarity_percentage:.2f}%\n")
            if similarity_percentage > highest_similarity:
                highest_similarity = similarity_percentage
                best_match = (detected_text, closest_part, similarity_percentage, orientation)
        if best_match:
            detected_text, closest_part, similarity_percentage, orientation = best_match
            print("OCR: Best Match:")
            print(f"OCR: Orientation: {orientation}")
            print(f"OCR: Detected Text: {detected_text}")
            print(f"OCR: Closest Part Number: {closest_part}")
            print(f"OCR: Similarity: {similarity_percentage:.2f}%")
    except Exception as e:
        print(e)

# =============================================================================
# Step 6. Main Execution: Run AI then OCR
# =============================================================================
if __name__ == "__main__":
    print("===== Starting AI Detection =====")
    run_ai_detection()
    
    # Wait long enough to ensure the AI pipeline is fully released.
    time.sleep(10)
    gc.collect()
    Gst.init(None)  # Reinitialize GStreamer if needed
    
    print("\n===== Starting OCR Processing on Captured Image =====")
    run_ocr()
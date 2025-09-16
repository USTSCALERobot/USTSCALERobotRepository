#!/usr/bin/env python3
import os
import subprocess
import sys
import time

# --- Configuration ---
HAILO_ENV_SCRIPT = "/home/scalepi/hailo-rpi5-examples/setup_env.sh"
HAILO_VENV_PATH = "/home/scalepi/hailo-rpi5-examples/venv_hailo_rpi5_examples/bin/activate"
OCR_SCRIPT = "/home/scalepi/hailo-rpi5-examples/basic_pipelines/Final/ocr.py"
# Detection file written by chipvision; expected format: full_image_path,cropped_image_path,x1,y1,x2,y2
DETECTION_FILE = "/home/scalepi/Desktop/savephototest/latest_detection.txt"
OCR_SAVE_FOLDER = "/home/scalepi/Desktop/OCR"

# --- Environment Activation ---
def activate_env():
    if os.getenv("HAILO_ENV_ACTIVATED") == "1":
        print("✅ Environment already activated.")
        return
    print("🔧 Activating environment...")
    command = (
        f"bash -c 'source {HAILO_ENV_SCRIPT} && "
        f"source {HAILO_VENV_PATH} && "
        f"export HAILO_ENV_ACTIVATED=1 && env'"
    )
    result = subprocess.run(command, shell=True, executable="/bin/bash",
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"❌ Error activating environment: {result.stderr}")
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        os.environ[key] = value
    print("✅ Environment activated successfully.")

# --- Read Detection File ---
def read_detection_file():
    if not os.path.exists(DETECTION_FILE):
        sys.exit("❌ Detection file not found. Ensure chip detection succeeded.")
    with open(DETECTION_FILE, "r") as f:
        data = f.read().strip().split(",")
    if len(data) < 6:
        sys.exit("❌ Detection file format error.")
    # Expected format: full_image_path, cropped_image_path, x1, y1, x2, y2
    image_path = data[0]
    bbox = ",".join(data[2:6])
    return image_path, bbox

# --- Run OCR ---
def run_ocr():
    image_path, bbox = read_detection_file()
    print(f"Running OCR on image: {image_path} with bbox: {bbox}")
    command = [
        "python3", OCR_SCRIPT,
        "--image", image_path,
        "--bbox", bbox,
        "--save_folder", OCR_SAVE_FOLDER
    ]
    print("Running OCR with command:", " ".join(command))
    result = subprocess.run(command)
    if result.returncode != 0:
        sys.exit(f"❌ OCR processing failed with return code {result.returncode}")
    print("✅ OCR processing completed.")

# --- Cleanup Environment ---
def cleanup_env():
    # For virtual environments, no explicit deactivation is usually needed.
    print("Cleaning up environment (if necessary).")

# --- Main Workflow ---
def main():
    activate_env()
    time.sleep(1)  # Allow environment to settle
    run_ocr()
    cleanup_env()

if __name__ == "__main__":
    main()
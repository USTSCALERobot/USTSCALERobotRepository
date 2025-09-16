import time
import os
from picamera2 import Picamera2, Preview

# Define the folder where images will be saved
save_folder = '/home/scalepi/Train_Images'

# Ensure the folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize the camera
picam2 = Picamera2()

# Start the camera preview to show the live feed
picam2.start_preview(Preview.QTGL)  # This will open a preview window

# Configure the camera for still image capture
picam2.start()

# Set number of images to capture
num_images = 1300
pause_duration = 0.25  # 0.25 second pause between captures
long_pause_interval = 100  # Every 150 images, add a longer pause
long_pause_duration = 5  # 5 seconds pause

# Capture 600 images with a 0.25-second pause between each
for i in range(num_images):
    # Define the filename for each image (e.g., image001.jpg, image002.jpg, etc.)
    filename = os.path.join(save_folder, f"image{i+1:03d}.jpg")
    
    # Capture the image and save it to the specified folder
    picam2.capture_file(filename)
    
    # Print a message after each capture (optional)
    print(f"Captured {filename}")
    
    # Wait for the specified pause time before taking the next picture
    time.sleep(pause_duration)
    
    # Every 150 images, add an additional 5-second pause
    if (i + 1) % long_pause_interval == 0:
        print(f"Taking a 5-second break after {i + 1} images...")
        time.sleep(long_pause_duration)

# Stop the camera preview window (optional, to free up resources)
picam2.stop_preview()

# Release the camera
picam2.stop()

print("Finished capturing 600 images.")

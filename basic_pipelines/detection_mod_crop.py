import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
from datetime import datetime
from hailo_rpi_common import (
    get_caps_from_pad,
    app_callback_class,
)
from detection_pipeline import GStreamerDetectionApp

# Folder path where processed frames and cropped images will be saved
SAVE_FOLDER = "/home/scalepi/Desktop/savephototest"

# Define a custom callback class for managing the detection pipeline
class user_app_callback_class(app_callback_class):
    def __init__(self, pipeline):
        super().__init__()
        self.stop_detection = False  # Flag to indicate when to stop detection
        self.use_frame = True  # Flag to indicate if frames should be processed
        self.pipeline = pipeline  # Reference to the GStreamer pipeline

# Function to extract raw frame data from a GStreamer buffer
def extract_raw_frame(buffer, width, height):
    success, map_info = buffer.map(Gst.MapFlags.READ)  # Map buffer for reading
    if not success:
        print("Failed to map buffer.")
        return None

    try:
        # Convert buffer data to a numpy array and reshape it to a frame
        raw_data = np.frombuffer(map_info.data, dtype=np.uint8)
        frame = raw_data.reshape((height, width, 3))  # Assuming RGB format
        print(f"Raw frame extracted with shape: {frame.shape}")
        return frame
    except Exception as e:
        print(f"Error extracting raw frame: {e}")
        return None
    finally:
        buffer.unmap(map_info)  # Unmap the buffer after use

# Function to crop the detected region from the frame
def crop_frame(frame, bbox):
    """
    Crop the image based on the bounding box coordinates.

    Parameters:
        frame (numpy.ndarray): The image frame (in RGB format).
        bbox (hailo.HailoBBox): Bounding box object with normalized values (0 to 1).

    Returns:
        numpy.ndarray: Cropped image.
    """
    try:
        # Frame dimensions
        height, width, _ = frame.shape

        # Convert normalized coordinates to pixel values
        x1 = int(bbox.xmin() * width)
        y1 = int(bbox.ymin() * height)
        x2 = int(bbox.xmax() * width)
        y2 = int(bbox.ymax() * height)

        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        # Crop the region from the frame
        cropped_image = frame[y1:y2, x1:x2]
        print(f"Cropped image size: {cropped_image.shape}")
        return cropped_image
    except Exception as e:
        print(f"Failed to crop frame: {e}")
        return None

# Callback function to process each frame in the GStreamer pipeline
def app_callback(pad, info, user_data):
    if user_data.stop_detection:  # Check if detection should stop
        print("Detection stopped. Dropping frames.")
        return Gst.PadProbeReturn.DROP

    buffer = info.get_buffer()  # Retrieve the buffer from the frame info
    if not buffer:
        print("Buffer is None, skipping frame.")
        return Gst.PadProbeReturn.OK

    # Extract frame format and dimensions
    format, width, height = get_caps_from_pad(pad)
    if not format or not width or not height:
        print(f"Invalid caps detected. Format: {format}, Width: {width}, Height: {height}")
        return Gst.PadProbeReturn.OK

    # Extract raw frame data
    frame = extract_raw_frame(buffer, width, height)
    if frame is None:
        print("Failed to extract raw frame from buffer.")
        return Gst.PadProbeReturn.OK

    print("Frame extracted successfully. Proceeding with detections.")

    # Perform object detection using Hailo SDK
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    for detection in detections:
        # Retrieve detection details
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        x1, y1, x2, y2 = bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()

        # Log detection information
        print(f"Detection - Label: {label}, BBox: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}), Confidence: {confidence:.2f}")

        # Stop detection if a condition is met
        if x2 >= 0.9:
            print(f"Condition met: x_max = {x2:.2f} >= 0.9")
            user_data.stop_detection = True  # Set stop flag

            # Save the full frame
            save_frame(frame)

            # Crop the detected region
            cropped_image = crop_frame(frame, bbox)
            if cropped_image is not None:
                # Save the cropped image
                save_cropped_image(cropped_image)

            # Stop the pipeline asynchronously
            GLib.idle_add(stop_pipeline, user_data.pipeline)
            return Gst.PadProbeReturn.DROP

    print("No detections meeting condition. Continuing.")
    return Gst.PadProbeReturn.OK

# Function to save a frame as an image
def save_frame(frame):
    try:
        # Convert the frame from RGB to BGR format for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not os.path.exists(SAVE_FOLDER):  # Create folder if it doesn't exist
            os.makedirs(SAVE_FOLDER)
        # Generate a unique filename with timestamp
        filename = os.path.join(SAVE_FOLDER, f"chip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, frame_bgr)  # Save the frame
        print(f"Frame saved at {filename}")
    except Exception as e:
        print(f"Failed to save frame: {e}")

# Function to save a cropped image
def save_cropped_image(cropped_image):
    try:
        if not os.path.exists(SAVE_FOLDER):  # Create folder if it doesn't exist
            os.makedirs(SAVE_FOLDER)
        # Generate a unique filename with timestamp
        filename = os.path.join(SAVE_FOLDER, f"cropped_chip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, cropped_image)  # Save the cropped image
        print(f"Cropped image saved at {filename}")
    except Exception as e:
        print(f"Failed to save cropped image: {e}")

# Function to stop the GStreamer pipeline
def stop_pipeline(pipeline):
    try:
        print("Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)  # Set pipeline state to NULL
        print("Pipeline stopped.")
    except Exception as e:
        print(f"Failed to stop pipeline: {e}")
    return False

# Main execution block
if __name__ == "__main__":
    Gst.init(None)  # Initialize GStreamer
    dummy_pipeline = Gst.Pipeline.new("dummy-pipeline")  # Create a dummy pipeline
    user_data = user_app_callback_class(dummy_pipeline)  # Instantiate user callback class
    app = GStreamerDetectionApp(app_callback, user_data)  # Initialize the detection app
    user_data.pipeline = app.pipeline  # Set the pipeline in user data

    try:
        app.run()  # Run the detection app
    except Exception as e:
        print(f"Application terminated: {e}")
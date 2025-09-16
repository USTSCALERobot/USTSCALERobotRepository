import gi
import os
import numpy as np
import cv2
from datetime import datetime
import hailo
from hailo_rpi_common import get_caps_from_pad, app_callback_class
from detection_pipeline import GStreamerDetectionApp

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

# Folder path where processed frames will be saved
SAVE_FOLDER = "/home/scalepi/Desktop/savephototest"

class UserAppCallback(app_callback_class):
    """Handles AI detection and stopping the pipeline when a chip is detected."""

    def __init__(self, pipeline, main_loop):
        super().__init__()
        self.stop_detection = False
        self.pipeline = pipeline
        self.main_loop = main_loop

    def stop_and_save(self, frame):
        """Stops detection, saves the image, and stops the pipeline."""
        save_frame(frame)
        GLib.idle_add(stop_pipeline, self.pipeline, self.main_loop)


def extract_raw_frame(buffer, width, height):
    """Extracts raw frame data from GStreamer buffer."""
    success, map_info = buffer.map(Gst.MapFlags.READ)
    if not success:
        print("❌ Failed to map buffer.")
        return None

    try:
        raw_data = np.frombuffer(map_info.data, dtype=np.uint8)
        return raw_data.reshape((height, width, 3))
    except Exception as e:
        print(f"❌ Error extracting raw frame: {e}")
        return None
    finally:
        buffer.unmap(map_info)


def app_callback(pad, info, user_data):
    """Handles frame-by-frame AI detection and stops the pipeline when a chip is found."""
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

        print(f"🔍 Detection - Label: {label}, BBox: ({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}), Confidence: {confidence:.2f}")

        if x2 >= 0.9:  # Condition to stop detection
            print(f"✅ Condition met: x_max = {x2:.2f} >= 0.9")
            user_data.stop_and_save(frame)
            return Gst.PadProbeReturn.DROP

    return Gst.PadProbeReturn.OK


def save_frame(frame):
    """Saves the detected frame."""
    try:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        os.makedirs(SAVE_FOLDER, exist_ok=True)
        filename = os.path.join(SAVE_FOLDER, f"chip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(filename, frame_bgr)
        print(f"📸 Frame saved at {filename}")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"❌ Failed to save frame: {e}")


def stop_pipeline(pipeline, main_loop):
    """Stops the GStreamer pipeline and exits the script cleanly."""
    try:
        print("🛑 Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        Gst.deinit()
        print("✅ Pipeline stopped.")

        if main_loop:
            main_loop.quit()

        os._exit(0)
    except Exception as e:
        print(f"❌ Failed to stop pipeline: {e}")


if __name__ == "__main__":
    Gst.init(None)
    main_loop = GLib.MainLoop()
    dummy_pipeline = Gst.Pipeline.new("dummy-pipeline")
    user_data = UserAppCallback(dummy_pipeline, main_loop)

    app = GStreamerDetectionApp(app_callback, user_data)
    user_data.pipeline = app.pipeline

    try:
        print("🚀 Running AI detection pipeline...")
        app.run()
        main_loop.run()
    except Exception as e:
        print(f"❌ Application terminated: {e}")
    finally:
        stop_pipeline(user_data.pipeline, user_data.main_loop)
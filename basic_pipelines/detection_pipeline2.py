import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import argparse
import multiprocessing
import numpy as np
import setproctitle
import cv2
import time
import hailo
from hailo_rpi_common import (
    get_default_parser,
    QUEUE,
    SOURCE_PIPELINE,
    DETECTION_PIPELINE,
    INFERENCE_PIPELINE_WRAPPER,
    USER_CALLBACK_PIPELINE,
    DISPLAY_PIPELINE,
    get_caps_from_pad,
    get_numpy_from_buffer,
    GStreamerApp,
    app_callback_class,
    dummy_callback,
)

class GStreamerDetectionApp(GStreamerApp):
    def __init__(self, app_callback, user_data):
        parser = get_default_parser()
        
        # Add additional arguments here
        parser.add_argument(
            "--network",
            default="yolov8s",
            choices=['yolov6n', 'yolov8s'],
            help="Which Network to use, default is yolov6n",
        )
        parser.add_argument(
            "--hef-path",
            #default="resources/Final.hef",
            default="resources/chip.hef",
            help="Path to HEF file",
        )
        parser.add_argument(
            "--labels-json",
            #default="resources/Final.json",
            default="default",
            help="Path to costume labels JSON file",
        )

        args = parser.parse_args()

        # Call the parent class constructor
        super().__init__(args, user_data)

        # Additional initialization code can be added here
        # Set Hailo parameters based on the model used
        self.batch_size = 2
        self.network_width = 640
        self.network_height = 640
        self.network_format = "RGB"
        nms_score_threshold = 0.3
        nms_iou_threshold = 0.45

        if args.hef_path is not None:
            self.hef_path = args.hef_path  # Use provided HEF path
        elif args.network == "yolov6n":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov6n.hef')
        elif args.network == "yolov8s":
            self.hef_path = os.path.join(self.current_path, '../resources/yolov8s_h8l.hef')
        else:
            assert False, "Invalid network type"

        # User-defined label JSON file
        self.labels_json = args.labels_json

        self.app_callback = app_callback

        self.thresholds_str = (
            f"nms-score-threshold={nms_score_threshold} "
            f"nms-iou-threshold={nms_iou_threshold} "
            f"output-format-type=HAILO_FORMAT_TYPE_FLOAT32"
        )

        # Set the process title
        setproctitle.setproctitle("IC Object Detection")

        # Create the pipeline
        self.create_pipeline()

    def get_pipeline_string(self):
        source_pipeline = SOURCE_PIPELINE(self.video_source)
        detection_pipeline = DETECTION_PIPELINE(
            hef_path=self.hef_path,
            batch_size=self.batch_size,
            labels_json=self.labels_json,
            additional_params=self.thresholds_str
        )
        user_callback_pipeline = USER_CALLBACK_PIPELINE()
        display_pipeline = DISPLAY_PIPELINE(video_sink=self.video_sink, sync=self.sync, show_fps=False)
        
        pipeline_string = (
            f'{source_pipeline} '
            f'! tee name=t '
            f't. {detection_pipeline} ! '
            f'{user_callback_pipeline} ! '
            f'{display_pipeline} '
            f't. ! queue ! appsink name=framesink'
        )
        print(pipeline_string)
        return pipeline_string

if __name__ == "__main__":
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()

    # Start the separate process for saving frames
    save_path = '/home/scalepi/hailo-rpi5-examples/OCR Images'
    frame_saver_process = multiprocessing.Process(target=save_frames, args=(user_data.raw_frame_queue, save_path))
    frame_saver_process.start()

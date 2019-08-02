# -*- coding: utf-8 -*-

import os

from imageai.Detection import VideoObjectDetection


assets = "./assets"
models = f"{assets}/models"
sample = f"{assets}/sample.mp4"
data = {
    "YOLOv3": {
        "class": "setModelTypeAsYOLOv3",
        "model": f"{models}/yolo.h5",
        "output": f"{assets}/sample.YOLOv3"
    },
    "TinyYOLOv3": {
        "class": "setModelTypeAsTinyYOLOv3",
        "model": f"{models}/yolo-tiny.h5",
        "output": f"{assets}/sample.TinyYOLOv3",
    },
    "RetinaNet": {
        "class": "setModelTypeAsRetinaNet",
        "model": f"{models}/resnet50_coco_best_v2.0.1.h5",
        "output": f"{assets}/sample.RetinaNet",
    },
}


for name, params in data.items():
    detector = VideoObjectDetection()
    getattr(detector, params["class"])()
    detector.setModelPath(params["model"])
    detector.loadModel()
    detector.detectObjectsFromVideo(
        input_file_path=sample,
        output_file_path=params["output"],
        frames_per_second=20,
        log_progress=True,
    )

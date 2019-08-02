# Video Tracking and Analysis with ImageAI

Experiment with the `VideoObjectDetection` class within the [ImageAI](http://imageai.org/) deep learning and computer vision library. 

## Dependencies

* ImageAI
* Keras
* OpenCV
* TensorFlow

## Installation

Install dependencies with Pipenv.

```bash
$ pipenv install
```

Download pre-trained models and place them in the `assets/models` directory.

* [RetinaNet](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5)
* [YOLOv3](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo.h5)
* [TinyYOLOv3](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/yolo-tiny.h5)

Run `experiment.py`. Note, this will take awhile.

```bash
$ python experiment.py
```

## Results

Here are the results for the above models. Note, videos were trimmed and converted into GIFs.

### YOLOv3

![YOLOv3](./assets/sample.YOLOv3.gif)

### TinyYOLOv3

![TinyYOLOv3](./assets/sample.TinyYOLOv3.gif)

### RetinaNet

![RetinaNet](./assets/sample.RetinaNet.gif)
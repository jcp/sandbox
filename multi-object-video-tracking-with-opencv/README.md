# Multi Object Video Tracking with OpenCV

Multi object tracking with the Open Source Computer Vision (OpenCV) library.

## Dependencies

* OpenCV
* Numpy

## Installation

Install dependencies with Pipenv.

```bash
$ pipenv install
```

Run `experiment.py`.

```bash
$ python experiment.py
```

## Results

Here are the results using background subtraction, morphological transformations and contour detection.

![Multi Object Video Tracking](./assets/example.gif)

### Morphological transformations

Before contours are detected, the gray scale image undergoes a series of morphological transformations. First, it's eroded and then dilated.  detection, each frame is treated with a series of morphological transformations.

#### BGR to Gray

![BGR to Gray](./assets/gray.jpg)

#### Closing

Frame is dilated and then eroded. This closes black points within foreground objects.

![Closing](./assets/closing.jpg)

#### Opening

Frame is eroded and then dilated. This removes noise around the foreground object.

![Opening](./assets/opening.jpg)

#### Dilation

Increase the size of the foreground object. This is useful for joining broken parts of an object.

![Dilation](./assets/dilation.jpg)

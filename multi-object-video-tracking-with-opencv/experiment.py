# -*- coding: utf-8 -*-

import cv2
import numpy as np


video = cv2.VideoCapture("assets/example.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()
fps = video.get(cv2.CAP_PROP_FPS)
height, width = (
    int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
)

output = cv2.VideoWriter(
    "assets/analyzed.avi",
    fourcc=cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    fps=fps,
    frameSize=(width, height),
)

while video.isOpened():
    ret, frame = video.read()

    if ret:
        gray = cv2.cvtColor(frame, code=cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Apply morphological transformations.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5, 5))
        closing = cv2.morphologyEx(fgmask, op=cv2.MORPH_CLOSE, kernel=kernel)
        opening = cv2.morphologyEx(closing, op=cv2.MORPH_OPEN, kernel=kernel)
        dilation = cv2.dilate(opening, kernel=kernel)

        # Apply threshold.
        _, thresh = cv2.threshold(
            dilation, thresh=127, maxval=255, type=cv2.THRESH_BINARY
        )

        # Find contours.
        contours, hierarchy = cv2.findContours(
            thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE
        )

        # Loop through each parent contour.
        for i in range(len(contours)):
            if hierarchy[0, i, 3] == -1:
                # Calculate the center of the contour.
                cnt = contours[i]
                M = cv2.moments(cnt)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Draw bounding box around contour.
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(
                    frame,
                    pt1=(x, y),
                    pt2=(x + w, y + h),
                    color=(102, 255, 0),
                    thickness=2,
                )

                # Mark the center of the contour.
                cv2.drawMarker(
                    frame,
                    position=(cx, cy),
                    color=(102, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=5,
                    thickness=1,
                    line_type=cv2.LINE_AA,
                )

        # Display and save video.
        cv2.imshow("Video", mat=frame)
        output.write(frame)

        # End when "q" is pressed.
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # End when video is over.
    else:
        break

video.release()
output.release()
cv2.destroyAllWindows()

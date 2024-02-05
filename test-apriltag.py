from pathlib import Path
import cv2
import numpy as np
import apriltag


video = cv2.VideoCapture(1)

video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while video.isOpened():
    check, frame = video.read()
    print(f"{check=}")
    print(f"{frame.shape=}")

    cv2.imshow("Capturing", frame)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    options = apriltag.DetectorOptions(families="tag16h5")

    detector = apriltag.Detector(options)

    detections = detector.detect(image)
    if detections:
        print(detections[0].homography)

    key = cv2.waitKey(1)
    if key == 27:
        break
    else:
        cv2.imshow("Please press the escape(esc) key to stop the video", frame)

video.release()

# print(detections[0].corners)
# print(detections[0].center)
# print(detections)

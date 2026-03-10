# Machine Exercise 1 -CO 543 Computer Vision
This repository demonstrates an AI-powered "Red Light, Green Light" game pipeline using webcam input. Motion detection is implemented with OpenCV-Python, which provides frame differencing, thresholding, and contour analysis to determine the movement state of the player. Additionally, to enhance accuracy of the game, the YOLOv8-nano model is integrated for real-time person detection, ensuring that motion is tracked only within detected player regions. The combination of OpenCV's image processing tools and YOLOv8's deep learning capabilities creates a robust machine vision system that overlays bounding boxes, motion values, game states directly on the video stream.

Below is the "Red Light, Green Light" video demonstration to showcase the pipeline in action.
(video: https://youtu.be/SkS0y2KKNiA)


#main.py

import numpy as np
import cv2
from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

def difference():
# Load the video file.
    cap = cv2.VideoCapture('Videos/goingDown.mp4')
    # cap = cv2.VideoCapture(-1)

    ret, img0 = cap.read()
    ret, img1 = cap.read()

    while cap.isOpened():
        ret, frame = cap.read()  # Read an frame from the video file.

        # If we cannot read any more frames from the video file, then exit.
        if not ret:
            break

        diff = cv2.subtract(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))

        # Move the data in img0 to img1. Uncomment this line for differencing from the first frame.
        img1 = img0
        ret, img0 = cap.read()  # Grab a new frame from the camera for img0.

        ret, diff = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        diff = cv2.medianBlur(diff,5)

        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=1)

        height = diff.shape[0]
        moments = cv2.moments(opening)
        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            print("X: {}, Y: {}".format(cX, height - cY))
            cv2.circle(opening, (cX, cY), 20, (255, 255, 0), -1)


        cv2.imshow('Difference', opening)  # Display the difference to the screen.

        # Close the script if q is pressed.
        # Note that the delay in cv2.waitKey affects how quickly the video will play on screen.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file, and close the GUI.
    cap.release()
    cv2.destroyAllWindows()

def difference_with_kalman():
# Load the video file.
    cap = cv2.VideoCapture('Videos/goingDown.mp4')
    # cap = cv2.VideoCapture(-1)

    timestep = 1/25  # Time between frames in the video.

    # Construct the Kalman Filter and initialize the variables.
    kalman = KalmanFilter(dim_x=4, dim_z=2)  # 4 state variables, 2 measured variables.
    kalman.x = np.array([0, 0, 0, 0])  # Initial state for the filter.
    kalman.F = np.array([[1,0,timestep,0],
                         [0,1,0,timestep],
                         [0,0,1,0],
                         [0,0,0,1]], np.float32)  # State Transition Matrix
    kalman.H = np.array([[1,0,0,0],[0,1,0,0]], np.float32)  # Measurement matrix.
    kalman.P = np.array([[1000,0,0,0],
                         [0,1000,0,0],
                         [0,0,1000,0],
                         [0,0,0,1000]], np.float32)  # Covariance Matrix
    kalman.R = np.array([[1,0],
                         [0,1]], np.float32)  # Measurement Noise
    kalman.Q = np.array([[1,0,0,0],
                         [0,1,0,0],
                         [0,0,100,0],
                         [0,0,0,100]], np.float32)  # Process Noise

    ret, img0 = cap.read()
    ret, img1 = cap.read()

    while cap.isOpened():
        ret, frame = cap.read()  # Read an frame from the video file.

        # If we cannot read any more frames from the video file, then exit.
        if not ret:
            break

        diff = cv2.subtract(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))

        # Move the data in img0 to img1. Uncomment this line for differencing from the first frame.
        img1 = img0
        ret, img0 = cap.read()  # Grab a new frame from the camera for img0.

        ret, diff = cv2.threshold(diff, 24, 255, cv2.THRESH_BINARY)
        diff = cv2.medianBlur(diff,5)

        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel, iterations=3)

        kalman.predict()

        height = diff.shape[0]
        moments = cv2.moments(opening)
        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            center = np.array([int(x),int(y)])
            print("X: {}, Y: {}".format(cX, height - cY))
            cv2.circle(img0, (cX, cY), 20, (255, 255, 0), -1)


        cv2.imshow('Difference', img0)  # Display the difference to the screen.

        # Close the script if q is pressed.
        # Note that the delay in cv2.waitKey affects how quickly the video will play on screen.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file, and close the GUI.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    difference()
    # difference_with_kalman()
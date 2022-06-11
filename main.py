#main.py

import numpy as np
import cv2

from filterpy.kalman import KalmanFilter, UnscentedKalmanFilter, MerweScaledSigmaPoints

def difference():
# Load the video file.
    cap = cv2.VideoCapture("Videos/bagVid.mp4")
    # cap = cv2.VideoCapture(-1)
    count = 2

    ret, img0 = cap.read()
    ret, img1 = cap.read()

    while cap.isOpened():

        # If we cannot read any more frames from the video file, then exit.
        if not ret:
            break

        sub = cv2.subtract(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        # Move the data in img0 to img1. Uncomment this line for differencing from the first frame.
        img1 = img0
        ret, img0 = cap.read()  # Grab a new frame from the camera for img0.
        if not ret:
            break

        ret, diff = cv2.threshold(sub, 24, 255, cv2.THRESH_BINARY)
        blur = cv2.medianBlur(diff,5)

        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel, iterations=2)

        height = diff.shape[0]
        moments = cv2.moments(opening)

        # sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
        # diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        # blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
        # opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
        
        
        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            # print("X: {}, Y: {}".format(cX, height - cY))
            # cv2.circle(opening, (cX, cY), 20, (255, 0, 255), -1)
            cv2.circle(img1, (cX, cY), 20, (255, 0, 255), -1)

        
        # cv2.imshow('Difference', sub)  # Display the difference to the screen.



        # imgtop = np.concatenate((sub, diff), axis=1)
        # imgtop = np.concatenate((imgtop, img0), axis=1)

        # imgbot = np.concatenate((blur, opening), axis=1)
        # imgbot = np.concatenate((imgbot, img1), axis=1)

        # imgcat = np.concatenate((imgtop, imgbot), axis=0)

        # scale_percent = 30 # percent of original size
        # width = int(imgcat.shape[1] * scale_percent / 100)
        # height = int(imgcat.shape[0] * scale_percent / 100)
        # dim = (width, height)
        
        # resize image
        # resized = cv2.resize(imgcat, dim, interpolation = cv2.INTER_AREA)

        # cv2.imshow('Bluyr', imgcat)
        count = count + 1
        # Close the script if q is pressed.  cv2.waitKey(20)
        # Note that the delay in cv2.waitKey affects how quickly the video will play on screen.
        if 0xFF == ord('q'):
            break

    # Release the video file, and close the GUI.
    cap.release()
    cv2.destroyAllWindows()
    print(count)

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

        height = diff.shape[0]
        moments = cv2.moments(opening)
        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            center = np.array([int(cX),int(cY)])
            print("X: {}, Y: {}".format(cX, height - cY))
            cv2.circle(img0, (cX, cY), 20, (255, 255, 0), -1)

            kalman.predict()
            center_ = (int(kalman.x[0]), int(kalman.x[1]))
            axis_lengths = (int(kalman.P_prior[0, 0]), int(kalman.P_prior[1, 1]))
            cv2.ellipse(diff, center_, axis_lengths, 0, 0, 360, color=(255, 0, 0))

            cv2.circle(diff, tuple(center), 1, (0,255,0), 2)  # Draw the center (not centroid!) of the ball.

            measured = np.array([center[0], center[1]], dtype="float32")
                # Update the Kalman filter with the current ball location if we have it.
            kalman.update(measured)
            print('Estimate:\t', np.int32(kalman.x))
            print('Variance:\t', np.diag(kalman.P))

        cv2.imshow('Difference', diff)  # Display the difference to the screen.

        # Close the script if q is pressed.
        # Note that the delay in cv2.waitKey affects how quickly the video will play on screen.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video file, and close the GUI.
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    difference()
    #difference_with_kalman()
import cv2
import glob
import numpy as np

pattern_size = (7, 9)
square_size = 0.02  # 2 cm

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (6,5,0)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob("webcamera-chess/*.jpg")  # Path to your calibration images

for fname in images:
    print(fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    print(f"{ret=}")
    # If found, add object points and image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        # cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        # cv2.imshow("img", img)
        # cv2.waitKey(500)  # Adjust the wait time for visualizing the images

# cv2.destroyAllWindows()


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

image = cv2.imread(fname)
undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
print(fname)
cv2.imshow("img", undistorted_img)
cv2.waitKey(5000)

print(f"{mtx=}")
print(f"{dist=}")

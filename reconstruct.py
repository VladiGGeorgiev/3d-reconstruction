import glob
import numpy as np
import cv2
import apriltag
import matplotlib.pyplot as plt


def calibrate_camera(chessboard_images_path):
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
    images = glob.glob(
        f"{chessboard_images_path}/*.jpg"
    )  # Path to your calibration images

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("Re-projection Error: {}".format(mean_error / len(objpoints)))

    return mtx, dist


def get_homography(images):
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)

    homographies = []

    for image in images:
        detections = detector.detect(image)
        if detections:
            homographies.append(detections[0].homography)

    return homographies


def find_matching_points(images):
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(images[0], None)
    kp2, des2 = sift.detectAndCompute(images[1], None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    pts1 = []
    pts2 = []
    good_matches = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.70 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(m)

    matched_img = cv2.drawMatches(
        images[0],
        kp1,
        images[1],
        kp2,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imshow("asd", matched_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return pts1, pts2, good_matches


def drawlines(img1, img2, lines, pts1, pts2):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines"""
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 3)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def find_epipolar_lines(images, pts1, pts2, F):

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1, 3)

    img1_drawn, _ = drawlines(images[0], images[1], lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1, 3)

    img2_drawn, _ = drawlines(images[1], images[0], lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img1_drawn)
    plt.subplot(122), plt.imshow(img2_drawn)
    plt.show()


def get_projection_matrices(K_matrix, F):
    E = np.dot(np.dot(K_matrix.T, F), K_matrix)

    # Perform SVD on the essential matrix
    U, S, Vt = np.linalg.svd(E)

    # Ensure that the determinant of U and Vt is positive
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Extract the rotation and translation
    R = np.dot(U, np.dot(np.diag([1, 1, 0]), Vt))
    T = U[:, 2]

    # Camera projection matrix P1 (assuming the first camera is at the origin)
    P1 = K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # Camera projection matrix P2
    P2 = K_matrix @ np.hstack((R, T.reshape(-1, 1)))
    return P1, P2


def main():
    K_matrix, dist = calibrate_camera(chessboard_images_path="webcamera-chess")

    images = [
        cv2.imread("images/image3.jpg", cv2.IMREAD_GRAYSCALE),
        cv2.imread("images/image4.jpg", cv2.IMREAD_GRAYSCALE),
    ]
    images[0] = cv2.undistort(images[0], K_matrix, dist, None, K_matrix)
    images[1] = cv2.undistort(images[1], K_matrix, dist, None, K_matrix)

    homographies = get_homography(images)
    print("H", f"{homographies[0]}")
    print("H", f"{homographies[1]}")

    points1, points2, matches = find_matching_points(images)
    pts1 = np.int32(points1)
    pts2 = np.int32(points2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    print(f"{K_matrix=}")
    print(f"{F=}")

    # epipolar_lines = find_epipolar_lines(images, pts1, pts2, F)

    P1, P2 = get_projection_matrices(K_matrix, F)
    print(f"{P1=}")
    print(f"{P2=}")

    points_3d_homogeneous = cv2.triangulatePoints(
        P1, P2, np.float32(points1).transpose(), np.float32(points2).transpose()
    )

    # Convert homogeneous coordinates to 3D coordinates
    points_3d = points_3d_homogeneous[:3, :] / points_3d_homogeneous[3, :]

    # Calculate Euclidean distances of each point to the position of the camera (0, 0, 0)
    depths = np.sqrt(np.sum(points_3d**2, axis=0))

    # Uncalibrated depth scale
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    scaled_depths = 255 * (depths - min_depth) / (max_depth - min_depth)
    print(f"{scaled_depths=}")

    # Visualize depth on the first image
    img = cv2.imread("images/image3.jpg")
    for i, p in enumerate(pts1):
        color = int(scaled_depths[i])
        img = cv2.circle(img, tuple(p), 5, [0, color, 0], -1)

    cv2.imshow("asd", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

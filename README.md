# Finding 3D Coordinates and Depth Points from Stereo Images

## Project Overview:

In this project, we aim to leverage stereo imaging techniques to reconstruct the 3D coordinates and depth information of corresponding points in a scene using a stereo camera setup. Stereo vision involves capturing images from two or more cameras with a known spatial relationship, allowing us to triangulate the 3D position of points in the captured images.

## Project Objectives:

1. **Camera Calibration**:
Calibrate the stereo camera system to obtain accurate intrinsic parameters for each camera. This involves determining the camera's focal length, principal point, and distortion coefficients.

2. **Undistort Images**:
The goal of the "Undistort Images" step is to correct for lens distortion in the captured images. Lens distortion can occur due to imperfections in camera lenses, leading to image distortions that can affect the accuracy of subsequent computer vision tasks. By applying undistortion, we ensure that the images used in stereo vision are free from lens-induced distortions, providing a more accurate basis for feature matching and depth calculation.

3. **Find Matching points**
The objective of this step is to find corresponding points between the undistorted and possibly rectified stereo images. Corresponding points are crucial for computing the disparity map, which, in turn, is used to calculate depth information in stereo vision. The Scale-Invariant Feature Transform (SIFT) algorithm is employed to identify distinctive keypoints and descriptors, and the FLANN library with the KDTree index is used for efficient nearest neighbor matching.

4. **Find extrinsic from corresponding points**:
The objective of this step is to compute the Fundamental Matrix and extract Rotation and Translation from it, a key element in stereo vision that encodes the epipolar geometry between two views. The Fundamental Matrix relates corresponding points in stereo images and is essential for rectification and triangulation processes.

5. **Triangulate points**
The primary objective of this step is to triangulate the corresponding points from the stereo images to estimate their 3D coordinates in the world space.

7. **Find depth from 3D**
The primary objective of this step is to calculate the depth information or the distance between the 3D point and the camera position for the triangulated 3D points. 
   
## Technologies and Tools:

Python with OpenCV for image processing, feature matching, and stereo vision algorithms.
NumPy for numerical operations and matrix manipulations.
Matplotlib or other visualization libraries for displaying the reconstructed 3D point cloud.

## Expected Outcomes:

By the end of the project, we anticipate obtaining a detailed 3D representation of the scene captured by the stereo cameras. This information can be valuable for various computer vision applications, including robotics, augmented reality, and depth-based object recognition.

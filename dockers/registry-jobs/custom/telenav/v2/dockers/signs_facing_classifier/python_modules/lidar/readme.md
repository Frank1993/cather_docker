##Module Description

  The purpose of this module is to allow for synchronization of images taken with OSC with VLP-16
LIDAR point clouds.
  The synchronization has 2 parts:
  * timestamp synchronization, to determine which LIDAR frame is closest to a given image.
  * LIDAR to camera calibration, to allow projecting point clouds onto images.

##Module Structure
TO be added...

##Things left to try:
  * understand how calibration works, try to make a more accurate camera calibration
  * compute the camera calibration error based on the re-projection error - this helps check the 
  camera calibration (https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html)
  * use PnP and RANSAC to compute the lidar/camera rotation and translation, instead of the existing optimization
  * use other error measuring instead such as L2 instead of mean squared error in the camera lidar calibration
   optimization
  * check if we can use KITTI calibration directly - can be done for monocular cameras, but we need more
  checker boards to fill a whole image.
  * rectify the images using the distortion coefficients before selecting lidar and camera point
   correspondences - tried with undistort, still no better results
  * think of a more accurate way of selecting lidar and camera point correspondences
  * run the projection of lidar points on an image with only a selection of points
  * attempt to calibrate the camera with more square grid images
  * attempt to calibrate the camera with asymmetric circle grid images
  * attempt to compute the extrinsic camera matrix based on computed intrinsic matrix and some info
   about the position of objects in the real world
  * check to see what is self-calibration
  * better understand the computer vision part related to cameras and calibration
   



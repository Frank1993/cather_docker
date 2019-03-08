import math

import cv2
import numpy
import yaml


class CameraInfo:
    def __init__(self):
        self.camera_name = None
        self.distortion_model = None
        self.width = 0
        self.height = 0
        self.K = None
        self.D = None
        self.R = None
        self.P = None

    @staticmethod
    def from_yaml(yaml_file):
        """ Reads the given yaml file and initializes the CameraInfo object. """

        with open(yaml_file, 'r') as stream:
            try:
                data = yaml.load(stream)
                print('yaml data: ', data)

                camera_info = CameraInfo()
                camera_info.camera_name = data['camera_name']
                camera_info.distortion_model = data['distortion_model']
                camera_info.width = data['image_width']
                camera_info.height = data['image_height']
                camera_info.K = data['camera_matrix']['data']
                camera_info.D = data['distortion_coefficients']['data']
                camera_info.R = data['rectification_matrix']['data']
                camera_info.P = data['projection_matrix']['data']

                return camera_info
            except yaml.YAMLError as exc:
                print(exc)


def mkmat(rows, cols, L):
    mat = numpy.matrix(L, dtype='float64')
    mat.resize((rows, cols))
    return mat


class PinholeCameraModel:

    """
    A pinhole camera is an idealized monocular camera.
    """

    def __init__(self):
        self.K = None
        self.D = None
        self.R = None
        self.P = None
        self.width = None
        self.height = None
        self.resolution = None
        self.mapx = None
        self.mapy = None

    @staticmethod
    def from_camera_info(camera_info):
        """
        :param camera_info: camera parameters
        :type camera_info:  CameraInfo

        Set the camera parameters from the :class:`CameraInfo` message.
        """
        cam_model = PinholeCameraModel()

        cam_model.K = mkmat(3, 3, camera_info.K)
        if camera_info.D:
            cam_model.D = mkmat(len(camera_info.D), 1, camera_info.D)
        else:
            cam_model.D = None

        cam_model.R = mkmat(3, 3, camera_info.R)
        cam_model.P = mkmat(3, 4, camera_info.P)
        cam_model.width = camera_info.width
        cam_model.height = camera_info.height
        cam_model.resolution = (camera_info.width, camera_info.height)

        return cam_model

    def rectify_image(self, raw):
        """
        :param raw:       input image
        :type raw:        :class:`CvMat` or :class:`IplImage`
        :param rectified: rectified output image

        Applies the rectification specified by camera parameters :math:`K` and and :math:`D` to image `raw` and writes
        the resulting image `rectified`.

        """

        self.mapx = numpy.ndarray(shape=(self.height, self.width, 1),
                           dtype='float32')
        self.mapy = numpy.ndarray(shape=(self.height, self.width, 1),
                           dtype='float32')
        cv2.initUndistortRectifyMap(self.K, self.D, self.R, self.P,
                (self.width, self.height), cv2.CV_32FC1, self.mapx, self.mapy)
        return cv2.remap(raw, self.mapx, self.mapy, cv2.INTER_CUBIC)

    def rectify_point(self, uv_raw):
        """
        :param uv_raw:    pixel coordinates
        :type uv_raw:     (u, v)

        Applies the rectification specified by camera parameters
        :math:`K` and and :math:`D` to point (u, v) and returns the
        pixel coordinates of the rectified point.
        """

        src = mkmat(1, 2, list(uv_raw))
        src.resize((1,1,2))
        dst = cv2.undistortPoints(src, self.K, self.D, R=self.R, P=self.P)
        return dst[0,0]

    def project_3d_to_pixel(self, point):
        """
        :param point:     3D point
        :type point:      (x, y, z)

        Returns the rectified pixel coordinates (u, v) of the 3D point,
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`projectPixelTo3dRay`.
        """
        src = mkmat(4, 1, [point[0], point[1], point[2], 1.0])
        dst = self.P * src
        x = dst[0,0]
        y = dst[1,0]
        w = dst[2,0]
        if w != 0:
            return (x / w, y / w)
        else:
            return (float('nan'), float('nan'))

    def project_pixel_to_3d_ray(self, uv):
        """
        :param uv:        rectified pixel coordinates
        :type uv:         (u, v)

        Returns the unit vector which passes from the camera center to through rectified pixel (u, v),
        using the camera :math:`P` matrix.
        This is the inverse of :meth:`project3dToPixel`.
        """
        x = (uv[0] - self.cx()) / self.fx()
        y = (uv[1] - self.cy()) / self.fy()
        norm = math.sqrt(x*x + y*y + 1)
        x /= norm
        y /= norm
        z = 1.0 / norm
        return (x, y, z)

    def get_delta_u(self, deltaX, Z):
        """
        :param deltaX:          delta X, in cartesian space
        :type deltaX:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta u, given Z and delta X in Cartesian space.
        For given Z, this is the inverse of :meth:`getDeltaX`.
        """
        fx = self.P[0, 0]
        if Z == 0:
            return float('inf')
        else:
            return fx * deltaX / Z

    def get_delta_v(self, deltaY, Z):
        """
        :param deltaY:          delta Y, in cartesian space
        :type deltaY:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta v, given Z and delta Y in Cartesian space.
        For given Z, this is the inverse of :meth:`getDeltaY`.
        """
        fy = self.P[1, 1]
        if Z == 0:
            return float('inf')
        else:
            return fy * deltaY / Z

    def get_delta_x(self, deltaU, Z):
        """
        :param deltaU:          delta u in pixels
        :type deltaU:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta X, given Z in cartesian space and delta u in pixels.
        For given Z, this is the inverse of :meth:`getDeltaU`.
        """
        fx = self.P[0, 0]
        return Z * deltaU / fx

    def get_delta_y(self, deltaV, Z):
        """
        :param deltaV:          delta v in pixels
        :type deltaV:           float
        :param Z:               Z, in cartesian space
        :type Z:                float
        :rtype:                 float

        Compute delta Y, given Z in cartesian space and delta v in pixels.
        For given Z, this is the inverse of :meth:`getDeltaV`.
        """
        fy = self.P[1, 1]
        return Z * deltaV / fy

    def full_resolution(self):
        """Returns the full resolution of the camera"""
        return self.resolution

    def intrinsic_matrix(self):
        """ Returns :math:`K`, also called camera_matrix in cv docs """
        return self.K

    def distortion_coeffs(self):
        """ Returns :math:`D` """
        return self.D

    def rotation_matrix(self):
        """ Returns :math:`R` """
        return self.R

    def projection_matrix(self):
        """ Returns :math:`P` """
        return self.P

    def cx(self):
        """ Returns x center """
        return self.P[0, 2]

    def cy(self):
        """ Returns y center """
        return self.P[1, 2]

    def fx(self):
        """ Returns x focal length """
        return self.P[0, 0]

    def fy(self):
        """ Returns y focal length """
        return self.P[1, 1]

    def Tx(self):
        """ Return the x-translation term of the projection matrix """
        return self.P[0, 3]

    def Ty(self):
        """ Return the y-translation term of the projection matrix """
        return self.P[1, 3]


if __name__ == '__main__':
    """ For testing purposes... """
    info = CameraInfo.from_yaml('/Users/mihaic7/Development/projects/telenav/lidar/data/calibration/cam_calibration/calib_data/oneplus5t_camera_info.yaml')
    cam_model = PinholeCameraModel.from_camera_info(info)
    print('camera matrix: ', cam_model.K)
    print('distortion coefficients: ', cam_model.D)
    print('rectification matrix: ', cam_model.R)
    print('projection matrix: ', cam_model.P)

    img = cv2.imread('/Users/mihaic7/Development/projects/telenav/lidar/data/calibration/cam_calibration/imgs/IMG_20181026_102239.jpg')
    rectified = cam_model.rectify_image(img)
    cv2.imwrite('/Users/mihaic7/Development/projects/telenav/lidar/data/calibration/cam_calibration/imgs/IMG_20181026_102239_rectified.jpg', rectified)

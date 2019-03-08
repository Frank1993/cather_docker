import argparse
import logging
import math
import os
import random

import commentjson as json
import numpy as np
import lidar.common.transformations as tf

from scipy.optimize import minimize
from apollo_python_common import log_util, io_utils
from lidar.common.virtual_camera import CameraInfo, PinholeCameraModel


class LidarToCamera:
    """ This class has the purpose of calculating the rotation and translation matrices needed to convert points from
    the LIDAR 3D coordinate system to the camera 2D coordinate system. """

    def __init__(self, camera_model, params, transform_file, err_threshold=45, max_iter=500):
        self.camera_model = camera_model
        self.params = params
        self.transform_file = transform_file
        self.err_threshold = err_threshold
        self.max_iter = max_iter

    def _cost_function(self, x0):
        # state_vars = [tx ty tz yaw pitch roll]
        state_vars = x0
        # logger.info('')
        # logger.info(state_vars)
        translation = [state_vars[0], state_vars[1], state_vars[2], 1.0]

        # euler_matrix format roll, pitch, yaw angles
        rotation_matrix = tf.euler_matrix(state_vars[5], state_vars[4], state_vars[3])
        rotation_matrix[:, 3] = translation

        error = 0
        for i in range(0, len(self.params['points'])):
            point = self.params['points'][i]
            input_uv = self.params['uvs'][i]

            # rotate the input point and project it to get uv space
            rotated_point = rotation_matrix.dot(point)
            uv = self.camera_model.project_3d_to_pixel(rotated_point)

            # calculate error between expected uv and calculated uv
            diff = np.array(uv) - np.array(input_uv)
            error = error + math.sqrt(np.sum(diff * diff))
        logger.info('cost function: ')
        logger.info("input_uv: {}".format(input_uv))
        logger.info("computed_uv: {}".format(uv))
        logger.info("uv_difference: {}".format(diff))
        logger.info("error: {}".format(error))

        return error

    def _save(self, transform, error):
        transform_dict = {'transform': transform, 'error': error}
        io_utils.json_dump(transform_dict, self.transform_file)

    def optimize(self):
        logger.info('Start optimization...')
        result = minimize(self._cost_function, self.params['initTransform'], args=(), bounds=self.params['bounds'],
                          method='SLSQP', options={'disp': True, 'maxiter': self.max_iter})

        # run till the optimization returns no success or if the obj function value >45
        while not result.success or result.fun > self.err_threshold:
            for i in range(0, len(self.params['initTransform'])):
                # choose random state vector from within the bounds
                self.params['initTransform'][i] = random.uniform(self.params['bounds'][i][0],
                                                                 self.params['bounds'][i][1])

            logger.info('')
            logger.info('Trying new starting point:')
            logger.info(self.params['initTransform'])
            result = minimize(self._cost_function, self.params['initTransform'], args=(), bounds=self.params['bounds'],
                              method='SLSQP', options={'disp': True, 'maxiter': self.max_iter})

        logger.info('Finished 2D-3D correspondences calculation..')
        logger.info('Final static transform :')
        logger.info(result.x)
        # value of the objective function
        logger.info('Error: ' + str(result.fun))
        self._save(list(result.x), result.fun)


def run(lidar_camera_cfg):
    cam_info = CameraInfo.from_yaml(os.path.join(lidar_camera_cfg['base_dir'], lidar_camera_cfg['camera_info']))
    cam_model = PinholeCameraModel.from_camera_info(cam_info)

    params_file = open(os.path.join(lidar_camera_cfg['base_dir'], lidar_camera_cfg['params_file']), 'r')
    params = json.load(params_file)

    transform_file = os.path.join(lidar_camera_cfg['base_dir'], lidar_camera_cfg['transform_file'])

    lidar_to_cam = LidarToCamera(cam_model, params, transform_file,
                                 err_threshold=lidar_camera_cfg['err_threshold'], max_iter=lidar_camera_cfg['max_iter'])
    lidar_to_cam.optimize()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    lidar_cam_cfg = io_utils.json_load(args.config)

    try:
        run(lidar_cam_cfg)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
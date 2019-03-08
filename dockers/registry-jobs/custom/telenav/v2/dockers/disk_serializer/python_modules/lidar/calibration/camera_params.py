import argparse
import logging
import math
import numpy as np

from apollo_python_common import log_util, io_utils


class CameraParams:
    def __init__(self, img_width, img_height, focal_length=0, fov=0, pixel_size=0):
        self.focal_length = focal_length / pixel_size
        self.fov = fov
        self.img_width = img_width
        self.img_heigth = img_height

    def _compute_intrinsics_with_fov(self):
        K = np.zeros((3, 3))
        print('received fov:', self.fov)
        focal_length = self.img_heigth / 2 / math.tan(self.fov / 2)
        print('in fov - focal length: ', focal_length)
        K[0, 0] = focal_length
        K[1, 1] = focal_length
        K[2, 2] = 1
        K[0, 2] = self.img_width / 2
        K[1, 2] = self.img_heigth / 2

        return K

    def _compute_intrinsics_with_focal(self):
        K = np.zeros((3, 3))

        print("focal length: ", self.focal_length)
        K[0, 0] = self.focal_length
        K[1, 1] = self.focal_length
        K[2, 2] = 1
        K[0, 2] = self.img_width / 2
        K[1, 2] = self.img_heigth / 2
        return K

    def run(self):
        print('intrinsics with fov: ', self._compute_intrinsics_with_fov())
        print('intrinsics with focal length: ', self._compute_intrinsics_with_focal())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    camera_params_cfg = io_utils.json_load(args.config)

    try:
        camera_params = CameraParams(camera_params_cfg['img_width'], camera_params_cfg['img_height'],
                                     camera_params_cfg['focal_length'], camera_params_cfg['fov'],
                                     camera_params_cfg['pixel_size'])
        camera_params.run()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
import argparse
import logging
import os

import cv2

import pandas as pd
import lidar.common.transformations as tf

from apollo_python_common import log_util, io_utils
from lidar.common.virtual_camera import CameraInfo, PinholeCameraModel
from lidar.data_handling import data_handler


class LidarCameraOverlayView:
    """ Loads images and lidar frames and overlays the lidar point cloud over the corresponding image """
    WINDOW_NAME = "lidar over image"

    def __init__(self, overlay_config):
        self._setup(overlay_config)

    def _setup(self, overlay_config):
        """ Sets up the internal fields of the class. """

        self.input_dir = overlay_config['input_dir']
        self.matched_df = pd.read_csv(overlay_config['matched_lidar_cam_file'])
        self.trans, self.rot = self._load_transform(overlay_config['transform_file'])
        cam_info = CameraInfo.from_yaml(overlay_config['camera_info'])
        self.camera_model = PinholeCameraModel.from_camera_info(cam_info)

    @staticmethod
    def _load_transform(transform_file):
        """ Loads the translation and rotation transform params. """
        transform_params = io_utils.json_load(transform_file)
        transform_lst = transform_params['transform']
        trans, rot = transform_lst[0:3], transform_lst[3:6]  # translation, rotation

        logger.info('translation: {}, rotation: {}'.format(trans, rot))

        return trans, rot

    def _display_overlay(self, img_path, frame_path):
        img = cv2.imread(img_path).copy()
        points = data_handler.load_lidar_frame(frame_path)

        print('translation: ', self.trans)
        print('rotation: ', self.rot)
        trans = tuple(self.trans) + (1, )
        rot = tuple(self.rot) + (1, )
        print('tupled translation: ', trans)
        print('tupled rotation: ', rot)

        rotation_matrix = tf.quaternion_matrix(rot)
        rotation_matrix[:, 3] = trans
        print('rotation_matrix: ', rotation_matrix)
        print('img shape: ', img.shape)

        for i in range(0, len(points) - 1):
            # convert to homogeneous coordinates
            point = [points[i][0], points[i][1], points[i][2], 1]  # TODO if conversion fails, add try/catch
            # apply rotation and translation
            rotated_point = rotation_matrix.dot(point)
            # project the point in 2d as a pixel
            uv = self.camera_model.project_3d_to_pixel(rotated_point)

            if 0 <= uv[0] <= img.shape[1] and 0 <= uv[1] <= img.shape[0]:
                cv2.circle(img, (int(uv[0]), int(uv[1])), 2, (0, 0, 255), 2)

        cv2.imshow(self.WINDOW_NAME, img)

    def run(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        for idx, row in self.matched_df.iterrows():
            img_path = os.path.join(self.input_dir, row['image_name'])
            frame_path = os.path.join(self.input_dir, row['lidar_frame_name'])
            logger.info('loading frame {} over image {}...'.format(frame_path, img_path))

            self._display_overlay(img_path, frame_path)

            key = cv2.waitKey()
            if key == 32:  # this should be space
                continue
            if key == ord('q'):
                break

        cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    overlay_cfg = io_utils.json_load(args.config)  # Load the config params

    try:
        overlay_view = LidarCameraOverlayView(overlay_cfg)
        overlay_view.run()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

import argparse
import logging
import os

import cv2

from apollo_python_common import log_util, io_utils


class ImagePointHandler:

    def __init__(self, input_img_file, points_file):
        self.point_list = []
        self.input_image = cv2.imread(input_img_file)
        self.points_file = points_file

    def _clicked_point(self, point):
        cv2.circle(self.input_image, point, 10, (0, 0, 255), 3)
        cv2.imshow("image", self.input_image)

    def _mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            point = (x, y)
            self.point_list.append(point)
            self._clicked_point(point)
            print(x, y)

    def _save_selected_points(self):
        file = open(self.points_file, 'w')
        for x, y in self.point_list:
            file.write("[{}, {}],\n".format(x, y))

        file.close()

    def run(self):
        print('image shape: ', self.input_image.shape)
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback("image", self._mouse_callback)
        clone = self.input_image.copy()

        while True:
            cv2.imshow("image", self.input_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self._save_selected_points()
            if key == ord('r'):
                self.input_image = clone
                clone = self.input_image.copy()
                self.point_list = []
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
    handler_config = io_utils.json_load(args.config)

    input_img_file = os.path.join(handler_config['working_dir'], handler_config['img_file'])
    points_file = os.path.join(handler_config['working_dir'], handler_config['points_file'])

    try:
        ip_handler = ImagePointHandler(input_img_file, points_file)
        ip_handler.run()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err
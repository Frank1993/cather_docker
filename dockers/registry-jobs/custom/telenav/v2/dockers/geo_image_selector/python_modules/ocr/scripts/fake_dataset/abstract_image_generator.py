import random

import cv2
import numpy as np
from apollo_python_common.image_utils.perspective_transformer import PerspectiveTransformer
from ocr.scripts.fake_dataset.generator_params import GeneratorParams as gp


class AbstractImageGenerator:

    def __init__(self, config):
        self.config = config

    def rotate_angle_z(self, img, ang):
        return PerspectiveTransformer(img).rotate_along_axis(phi=ang, dx=0)

    def rotate_image_x(self, image, angle):
        height, width, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return result

    def random_rotate(self, img, x_rotation_angle, z_rotation_angle):
        img = self.rotate_image_x(img, x_rotation_angle)
        img = self.rotate_angle_z(img, z_rotation_angle)
        return img

    def get_crop_points(self, img, max_img_width, max_img_height):

        height, width, _ = img.shape
        img_cols_first = np.transpose(img, (1, 0, 2))

        tl_x = 0
        tl_y = 0
        br_x = max_img_width
        br_y = max_img_height

        for index, col in enumerate(img_cols_first):
            if ([self.config[gp.COLOR_THRESH], 0, 0] <= col).all(1).any():
                tl_x = index
                break

        for index, col in enumerate(reversed(img_cols_first)):
            if ([self.config[gp.COLOR_THRESH], 0, 0] <= col).all(1).any():
                br_x = width - index
                break

        for index, col in enumerate(img):
            if ([self.config[gp.COLOR_THRESH], 0, 0] <= col).all(1).any():
                tl_y = index
                break

        for index, col in enumerate(reversed(img)):
            if ([self.config[gp.COLOR_THRESH], 0, 0] <= col).all(1).any():
                br_y = height - index
                break

        width_delta = (br_x - tl_x) * random.uniform(0.1, 0.25)
        height_delta = (br_y - tl_y) * random.uniform(0.2, 0.25)

        tl_x -= width_delta
        tl_y -= height_delta
        br_x += width_delta
        br_y += height_delta

        tl_x = max(tl_x, 0)
        tl_y = max(tl_y, 0)
        br_x = min(br_x, max_img_width)
        br_y = min(br_y, max_img_height)

        return int(tl_x), int(tl_y), int(br_x), int(br_y)

    def add_noise(self, img):
        noise = np.random.random(img.shape) * self.config[gp.NOISE_AMPLIFIER]
        img += noise.astype(np.uint8)
        return img

    def add_blur(self, img):
        blur_level = random.choice(range(self.config[gp.MIN_BLUR_LEVEL], self.config[gp.MAX_BLUR_LEVEL] + 1, 2))
        img = cv2.GaussianBlur(img, (blur_level, blur_level), 0)
        return img

    def adjust_gamma(self, image, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def generate_img(self, text):
        pass

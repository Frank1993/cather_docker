import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
import random
import os
from glob import glob
from random import randint
import PIL

from ocr.scripts.perspective_transformer import PerspectiveTransformer
from ocr.scripts.background_generator import BackgroundGenerator


class ImageGenerator:
    LEFT_OFFSET = 100
    UP_OFFSET = 25
    YELLOW_BACKROUND_PROB = 0.075
    BLUR_LEVEL = 3
    NOISE_AMPLIFIER = 10
    CANVAS_MULTIPLIER = 3
    COLOR_THRESH = 200
    HEIGHT_DELTA = 0
    WIDTH_DELTA = 0
    MIN_FONT_SIZE = 20
    MAX_FONT_SIZE = 60
    MIN_X_ROTATION_ANGLE = -5
    MAX_X_ROTATION_ANGLE = 5
    MIN_Z_ROTATION_ANGLE = -30
    MAX_Z_ROTATION_ANGLE = 30

    def __init__(self, width, height, resources_path):
        self.background_generator = BackgroundGenerator(os.path.join(resources_path, "backgrounds"),
                                                        os.path.join(resources_path, "yellow_backgrounds"))
        self.target_width = width
        self.target_height = height

        self.width = self.target_width * self.CANVAS_MULTIPLIER
        self.height = self.target_height * self.CANVAS_MULTIPLIER

        self.fonts_paths = glob(os.path.join(resources_path, "fonts") + "/*")

    def __rotate_angle_z(self, img, ang):
        return PerspectiveTransformer(img).rotate_along_axis(phi=ang, dx=0)

    def __rotate_image_x(self, image, angle):
        height, width, _ = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        result = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
        return result

    def __random_rotate(self, img, x_rotation_angle=None, z_rotation_angle=None):

        if x_rotation_angle is None:
            x_rotation_angle = np.random.randint(-5, 5)

        if z_rotation_angle is None:
            z_rotation_angle = np.random.randint(-30, 30)

        img = self.__rotate_image_x(img, x_rotation_angle)
        img = self.__rotate_angle_z(img, z_rotation_angle)

        return img

    def __get_image_with_text(self, pil_im, text, font_path, font_size, bold_range):

        draw = ImageDraw.Draw(pil_im)

        font = ImageFont.truetype(font_path, font_size)

        draw.text((self.LEFT_OFFSET, self.UP_OFFSET), text, font=font, fill=(255, 0, 0, 255))

        for index in bold_range:
            draw.text((self.LEFT_OFFSET + index, self.UP_OFFSET + index), text, font=font, fill=(255, 0, 0, 255))
            draw.text((self.LEFT_OFFSET - index, self.UP_OFFSET - index), text, font=font, fill=(255, 0, 0, 255))
        img = np.array(pil_im)

        return img

    def __get_crop_points(self, img):

        height, width, _ = img.shape
        img_cols_first = np.transpose(img, (1, 0, 2))

        tl_x = 0
        tl_y = 0
        br_x = self.width
        br_y = self.height

        for index, col in enumerate(img_cols_first):
            if ([self.COLOR_THRESH, 0, 0] <= col).all(1).any():
                tl_x = index
                break

        for index, col in enumerate(reversed(img_cols_first)):
            if ([self.COLOR_THRESH, 0, 0] <= col).all(1).any():
                br_x = width - index
                break

        for index, col in enumerate(img):
            if ([self.COLOR_THRESH, 0, 0] <= col).all(1).any():
                tl_y = index
                break

        for index, col in enumerate(reversed(img)):
            if ([self.COLOR_THRESH, 0, 0] <= col).all(1).any():
                br_y = height - index
                break

        self.WIDTH_DELTA = (br_x - tl_x) * random.uniform(0, 0.25)
        self.HEIGHT_DELTA = (br_y - tl_y) * random.uniform(0, 0.25)
        
        tl_x -= self.WIDTH_DELTA
        tl_y -= self.HEIGHT_DELTA
        br_x += self.WIDTH_DELTA
        br_y += self.HEIGHT_DELTA

        tl_x = max(tl_x, 0)
        tl_y = max(tl_y, 0)
        br_x = min(br_x, self.width)
        br_y = min(br_y, self.height)

        return tl_x, tl_y, br_x, br_y

    def __convert_text_color(self, img):

        delta = random.choice(np.arange(0.7, 0.9, 0.01))
        r = delta + random.choice(np.arange(-0.05, 0.05, 0.01))
        g = delta + random.choice(np.arange(-0.05, 0.05, 0.01))
        b = delta + random.choice(np.arange(-0.05, 0.05, 0.01))

        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        img[np.where((img >= [100, 0, 0]).all(axis=2))] = [r, g, b]

        r_avg = int(np.median(img[:, :, 0]))
        g_avg = int(np.median(img[:, :, 1]))
        b_avg = int(np.median(img[:, :, 2]))

        img[np.where((img <= [50, 50, 50]).all(axis=2))] = [r_avg, g_avg, b_avg]

        return img

    def __convert_text_color_yellow(self, img):

        delta = random.choice(np.arange(0.2, 0.4, 0.01))
        r = delta + random.choice(np.arange(-0.05, 0.05, 0.01))
        g = delta + random.choice(np.arange(-0.05, 0.05, 0.01))
        b = delta + random.choice(np.arange(-0.05, 0.05, 0.01))

        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        img[np.where((np.bitwise_and(img >= [165, 0, 0], img <= [255, 100, 255])).all(axis=2))] = [r, g, b]

        return img

    def __add_noise(self, img):
        noise = np.random.random(img.shape) * self.NOISE_AMPLIFIER
        img += noise.astype(np.uint8)

        return img

    def __add_blur(self, img):
        img = cv2.GaussianBlur(img, (self.BLUR_LEVEL, self.BLUR_LEVEL), 0)
        return img

    def generate_img(self, text):

        is_yellow = np.random.random() <= self.YELLOW_BACKROUND_PROB

        real_background = self.background_generator.get_random_background(is_yellow, self.width, self.height)
        black_background = self.background_generator.get_black_background(self.width, self.height)

        font_size = np.random.randint(self.MIN_FONT_SIZE, self.MAX_FONT_SIZE)
        font_path = random.choice(self.fonts_paths)
        bold_range = range(1, randint(0, 2))

        img = self.__get_image_with_text(real_background, text, font_path, font_size, bold_range)
        black_img = self.__get_image_with_text(black_background, text, font_path, font_size, bold_range)

        x_rotation_angle = np.random.randint(self.MIN_X_ROTATION_ANGLE, self.MAX_X_ROTATION_ANGLE)
        z_rotation_angle = np.random.randint(self.MIN_Z_ROTATION_ANGLE, self.MAX_Z_ROTATION_ANGLE)

        img = self.__random_rotate(img, x_rotation_angle, z_rotation_angle)
        black_img = self.__random_rotate(black_img, x_rotation_angle, z_rotation_angle)

        tl_x, tl_y, br_x, br_y = self.__get_crop_points(black_img)

        img = self.__convert_text_color_yellow(img) if is_yellow else self.__convert_text_color(img)

        img = self.__add_noise(img)
        img = self.__add_blur(img)

        img = np.asarray(PIL.Image.fromarray(img).crop((tl_x, tl_y, br_x, br_y)).resize((self.target_width,
                                                                                         self.target_height)))

        return img

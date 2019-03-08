import os
import random
from glob import glob

import PIL
import apollo_python_common.image as image_api
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ocr.scripts.fake_dataset.abstract_image_generator import AbstractImageGenerator
from ocr.scripts.fake_dataset.background_generator import BackgroundGenerator
from ocr.scripts.fake_dataset.generator_params import GeneratorParams as gp


class TrafficSignsImageGenerator(AbstractImageGenerator):

    def __init__(self, config):
        super().__init__(config)
        self.background_generator = BackgroundGenerator(os.path.join(self.config[gp.RESOURCES_PATH], "backgrounds"))
        self.margin_bg = BackgroundGenerator(os.path.join(self.config[gp.RESOURCES_PATH], "margin_backgrounds"))
        self.fonts_paths = glob(os.path.join(self.config[gp.RESOURCES_PATH], "fonts") + "/*")

    def __get_image_with_text(self, pil_im, text, font_path, font_size, bold_range, left_offset):

        draw = ImageDraw.Draw(pil_im)

        font = ImageFont.truetype(font_path, font_size)

        draw.text((self.config[gp.LEFT_OFFSET] + left_offset, self.config[gp.UP_OFFSET]),
                  text, font=font, fill=(255, 0, 0, 255))

        for index in bold_range:
            draw.text((self.config[gp.LEFT_OFFSET] + index, self.config[gp.UP_OFFSET] + index),
                      text, font=font, fill=(255, 0, 0, 255))
            draw.text((self.config[gp.LEFT_OFFSET] - index, self.config[gp.UP_OFFSET] - index),
                      text, font=font, fill=(255, 0, 0, 255))

        return np.array(pil_im)

    def __convert_rotation_background_color(self, img, background_pixel_value):
        img[np.where((img <= [50, 50, 50]).all(axis=2))] = background_pixel_value
        return img

    def __convert_text_color(self, img, background_pixel_value):

        img = self.__convert_rotation_background_color(img, background_pixel_value)

        delta = random.choice(np.arange(0, 0.1, 0.01))
        r = delta + random.choice(np.arange(-0.0, 0.1, 0.01))
        g = delta + random.choice(np.arange(-0.0, 0.1, 0.01))
        b = delta + random.choice(np.arange(-0.0, 0.1, 0.01))

        r = int(r * 255)
        g = int(g * 255)
        b = int(b * 255)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower_red = np.array([0, 10, 10])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        mask = mask0 + mask1

        img[np.where(mask == 255)] = [r, g, b]

        return img

    def __get_image(self, text, real_background, black_background, font_path, font_size, bold_range, left_offset):
        img = self.__get_image_with_text(real_background, text, font_path, font_size, bold_range, left_offset)
        black_img = self.__get_image_with_text(black_background, text, font_path, font_size, bold_range, left_offset)

        return img, black_img

    def __add_border(self, img, tl_x, tl_y, br_x, br_y, text):

        tl_x, tl_y, br_x, br_y = int(tl_x), int(tl_y), int(br_x), int(br_y)

        height, width, _ = img.shape

        border_delta_range = range(5, 10)
        border_thickness_range = range(2, 4)

        border_delta = random.choice(border_delta_range)
        border_thickness = random.choice(border_thickness_range)

        is_yellow = text == "no\nturns"
        if is_yellow:
            color = (random.choice(range(200, 255)), random.choice(range(200, 255)), random.choice(range(0, 10)))
        else:
            color = (random.choice(range(0, 30)), random.choice(range(0, 30)), random.choice(range(0, 30)))

        border_tl_x = tl_x + border_delta
        border_tl_y = tl_y + border_delta
        border_br_x = br_x - border_delta
        border_br_y = br_y - border_delta

        img = cv2.line(img, (border_tl_x, border_tl_y), (border_tl_x, border_br_y), color, border_thickness)
        img = cv2.line(img, (border_tl_x, border_tl_y), (border_br_x, border_tl_y), color, border_thickness)
        img = cv2.line(img, (border_tl_x, border_br_y), (border_br_x, border_br_y), color, border_thickness)
        img = cv2.line(img, (border_br_x, border_br_y), (border_br_x, border_tl_y), color, border_thickness)

        return img

    def __add_margins(self, original_img):
        img = PIL.Image.fromarray(original_img)
        img_width, img_height = img.size

        bg_height = int(img_height * random.choice(np.arange(0.85, 1.25, 0.05)))
        bg_width = int(img_width * random.choice(np.arange(0.85, 1.25, 0.05)))

        if bg_height == 0 or bg_width == 0:
            print("error:", img_width, img_height, bg_width, bg_height, original_img.shape)
            bg_height, bg_width = 500, 500

        real_background = self.margin_bg.get_random_background_stretched(bg_width, bg_height)

        new_size = real_background.size
        old_size = img.size

        real_background.paste(img, ((new_size[0] - old_size[0]) // 2,
                                    (new_size[1] - old_size[1]) // 2))

        return np.asarray(real_background).copy()

    def generate_img(self, original_text):
        texts = [" ", " "] + original_text.upper().split("\n") + [" ", " "]  # to fix cropping pb at high X rotations
        max_len = max([len(t) for t in texts])

        real_background = self.background_generator.get_random_background(self.config[gp.LINE_IMG_WIDTH],
                                                                          self.config[gp.LINE_IMG_HEIGHT])
        black_background = self.background_generator.get_black_background(self.config[gp.LINE_IMG_WIDTH],
                                                                          self.config[gp.LINE_IMG_HEIGHT])

        font_size = np.random.randint(self.config[gp.MIN_FONT_SIZE], self.config[gp.MAX_FONT_SIZE])
        font_path = random.choice(self.fonts_paths)
        bold_range = range(0)

        imgs = []
        black_imgs = []
        for i, text in enumerate(texts):
            left_offset = 1 / len(text) * max_len * 20
            img, black_img = self.__get_image(text, real_background.copy(), black_background.copy(), \
                                              font_path, font_size, bold_range, left_offset)
            imgs.append(img)
            black_imgs.append(black_img)

        big_img = np.concatenate(imgs)
        big_black_img = np.concatenate(black_imgs)

        random_color = random.choice(range(255))
        background_pixel_value = [random_color, random_color, random_color]

        x_rotation_angle = np.random.randint(self.config[gp.MIN_X_ROTATION_ANGLE],
                                             self.config[gp.MAX_X_ROTATION_ANGLE])
        z_rotation_angle = np.random.randint(self.config[gp.MIN_Z_ROTATION_ANGLE],
                                             self.config[gp.MAX_Z_ROTATION_ANGLE])

        big_img = self.random_rotate(big_img, x_rotation_angle, z_rotation_angle)
        big_black_img = self.random_rotate(big_black_img, x_rotation_angle, z_rotation_angle)

        max_width, max_height, _ = big_black_img.shape
        tl_x, tl_y, br_x, br_y = self.get_crop_points(big_black_img, max_width, max_height)

        big_img = self.__convert_text_color(big_img, background_pixel_value)

        big_img = self.__add_border(big_img, tl_x, tl_y, br_x, br_y, original_text)

        big_img = self.add_noise(big_img)
        big_img = self.add_blur(big_img)
        big_img = self.adjust_gamma(big_img, gamma=random.uniform(0.8, 1.5))

        big_img = np.asarray(PIL.Image.fromarray(big_img).crop((tl_x, tl_y, br_x, br_y)))

        big_img = self.__add_margins(big_img)
        big_img, _, _ = image_api.resize_image_fill(big_img, self.config[gp.FINAL_HEIGHT], self.config[gp.FINAL_WIDTH],
                                                    3)

        return big_img
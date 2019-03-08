import os
import random
from glob import glob
from random import randint

import PIL
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ocr.scripts.fake_dataset.abstract_image_generator import AbstractImageGenerator
from ocr.scripts.fake_dataset.background_generator import BackgroundGenerator
from ocr.scripts.fake_dataset.generator_params import GeneratorParams as gp


class SignpostImageGenerator(AbstractImageGenerator):
    
    def __init__(self, config):
        super().__init__(config)
        self.bg_gen = BackgroundGenerator(os.path.join(self.config[gp.RESOURCES_PATH], "backgrounds"))
        self.yellow_bg_gen = BackgroundGenerator(os.path.join(self.config[gp.RESOURCES_PATH], "yellow_backgrounds"))

        self.final_width = self.config[gp.FINAL_WIDTH]
        self.final_height = self.config[gp.FINAL_HEIGHT]

        self.canvas_width = self.final_width * self.config[gp.CANVAS_MULTIPLIER]
        self.canvas_height = self.final_height * self.config[gp.CANVAS_MULTIPLIER]

        self.fonts_paths = glob(os.path.join(self.config[gp.RESOURCES_PATH], "fonts") + "/*")

    def __get_image_with_text(self, pil_im, text, font_path, font_size, bold_range):

        draw = ImageDraw.Draw(pil_im)

        font = ImageFont.truetype(font_path, font_size)

        draw.text((self.config[gp.LEFT_OFFSET], self.config[gp.UP_OFFSET]), text, font=font, fill=(255, 0, 0, 255))

        for index in bold_range:
            draw.text((self.config[gp.LEFT_OFFSET] + index, self.config[gp.UP_OFFSET] + index), 
                      text, font=font, fill=(255, 0, 0, 255))
            draw.text((self.config[gp.LEFT_OFFSET] - index, self.config[gp.UP_OFFSET] - index), 
                      text, font=font, fill=(255, 0, 0, 255))

        return np.array(pil_im)

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

    def generate_img(self, text):

        is_yellow = np.random.random() <= self.config[gp.YELLOW_BACKROUND_PROB]

        chosen_bg_gen = self.yellow_bg_gen if is_yellow else self.bg_gen
        real_background = chosen_bg_gen.get_random_background(self.canvas_width, self.canvas_height)
        black_background = self.bg_gen.get_black_background(self.canvas_width, self.canvas_height)

        font_size = np.random.randint(self.config[gp.MIN_FONT_SIZE], self.config[gp.MAX_FONT_SIZE])
        font_path = random.choice(self.fonts_paths)
        bold_range = range(1, randint(0, 2))

        img = self.__get_image_with_text(real_background, text, font_path, font_size, bold_range)
        black_img = self.__get_image_with_text(black_background, text, font_path, font_size, bold_range)

        x_rotation_angle = np.random.randint(self.config[gp.MIN_X_ROTATION_ANGLE], self.config[gp.MAX_X_ROTATION_ANGLE])
        z_rotation_angle = np.random.randint(self.config[gp.MIN_Z_ROTATION_ANGLE], self.config[gp.MAX_Z_ROTATION_ANGLE])

        img = self.random_rotate(img, x_rotation_angle, z_rotation_angle)
        black_img = self.random_rotate(black_img, x_rotation_angle, z_rotation_angle)

        tl_x, tl_y, br_x, br_y = self.get_crop_points(black_img,self.canvas_width,self.canvas_height)

        img = self.__convert_text_color_yellow(img) if is_yellow else self.__convert_text_color(img)

        img = self.add_noise(img)
        img = self.add_blur(img)

        img = np.asarray(PIL.Image.fromarray(img).crop((tl_x, tl_y, br_x, br_y)).resize((self.final_width,
                                                                                         self.final_height)))

        return img

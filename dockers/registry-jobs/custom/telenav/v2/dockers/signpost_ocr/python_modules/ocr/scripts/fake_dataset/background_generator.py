import cv2
import random
import numpy as np
import PIL
from glob import glob
from sklearn.utils import shuffle


class BackgroundGenerator:

    def __init__(self, backgrounds_base_path):
        self.backgrounds = self.__get_backgrounds(backgrounds_base_path)

    def __get_backgrounds(self, path):
        backgrounds_paths = glob(path + "/*")
        backgrounds = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in backgrounds_paths]
        backgrounds_augm_horiz = [cv2.flip(img, 1) for img in backgrounds]
        backgrounds_augm_vert = [cv2.flip(img, 0) for img in backgrounds]
        backgrounds_augm_both = [cv2.flip(cv2.flip(img, 0), 1) for img in backgrounds]

        backgrounds = backgrounds + backgrounds_augm_horiz + backgrounds_augm_vert + backgrounds_augm_both

        return shuffle(backgrounds, random_state=0)

    def get_random_background(self, width, height):
        bg = random.choice(self.backgrounds)
        bg_height, bg_width, _ = bg.shape

        scale_width = width // bg_width + 1
        scale_height = height // bg_height + 1

        horiz_line = np.hstack([bg] * scale_width)
        bg = np.vstack([horiz_line] * scale_height)
        pil_im = PIL.Image.fromarray(bg).resize((width, height))

        return pil_im

    def get_black_background(self, width, height):
        return PIL.Image.fromarray(np.zeros((height, width, 3)).astype(np.uint8))

    def get_random_background_stretched(self, width, height):
        bg = random.choice(self.backgrounds)
        return PIL.Image.fromarray(bg).resize((width, height))
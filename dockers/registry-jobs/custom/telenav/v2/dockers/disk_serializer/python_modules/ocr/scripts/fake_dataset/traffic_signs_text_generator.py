import pandas as pd
import numpy as np
from itertools import chain
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import itertools
import random

from ocr.scripts.fake_dataset.abstract_text_generator import AbstractTextGenerator
from ocr.scripts.fake_dataset.poi_text_generator import PoiTextGenerator

class TrafficSignsTextGenerator(AbstractTextGenerator):
    
    DAY_TEMPLATES = ["MON-FRI","MON-SAT","MONDAY-FRIDAY","SCHOOL DAYS","FRI-SUN"]
    MAX_NR_FREE_TEXT_LINES = 3
    
    
    ONE_LINE_TIME_TEMPLATES = ["{} AM - {} PM","{} AM - {} AM","{} PM - {} PM",
                               "{} PM - {} AM","{} - {} AM","{} - {} PM"]
    TWO_LINE_TIME_TEMPLATES = ["{} AM \nto {} PM","{} PM \nto {} PM"]
    MAX_NR_HOUR_LINES = 2
    
    SIGN_TEMPLATES = [  "begin\none\nway",
                        "end\none\nway",
                        "end\none way\ntraffic",
                        "begin\ntwo\nway\ntraffic",
                        "road\nclosed",
                        "road closed\nto\nthru traffic",
                        "residential",
                        "when\nflashing",
                        "radar\nenforced",
                        "no\nturn\non\nred",
                        "no\nturn\non red",
                        "left lane\nmust\nturn left",
                        "right lane\nmust\nturn right",
                        "left\nlane",
                        "right\nlane",
                        "center\nlane",
                        "no\nturns",
                        "right lane\nmust\nexit",
                        "left lane\nmust\nenter ramp",
                        "no\nturns",
                        "begins",
                        "left turn signal",
                        "turn right only"]
                               
    def __init__(self,resources_path,nr_texts):
        self.free_text_templates = PoiTextGenerator(resources_path,nr_texts_per_length=10000,char_limit=16).generate_texts()
        self.nr_texts = nr_texts

    def __random_prob_choice(self, upper_thresh):
        return random.uniform(0,1) <= upper_thresh

    def __generate_free_sample_text(self):
        nr_lines = random.choice(range(1,self.MAX_NR_FREE_TEXT_LINES+1))
        text_lines = [random.choice(self.free_text_templates) for _ in range(nr_lines)]
        return "\n".join(text_lines)
    
    def __generate_day_text(self):
        return random.choice(self.DAY_TEMPLATES)
               
    def __generate_hour_text(self):

        hours = list(np.arange(0,24))
        minutes = ["00","05"] + [f"{t}" for t in np.arange(10,60,5)]

        use_two_line_template = self.__random_prob_choice(0.5)

        nr_templates = 1 if use_two_line_template else random.choice(range(1,self.MAX_NR_HOUR_LINES+1))

        text_lines = []

        for i in range(nr_templates):

            with_minutes = self.__random_prob_choice(0.5)

            if with_minutes:
                hour_templates = "{}:{}".format(random.choice(hours),random.choice(minutes)), \
                                 "{}:{}".format(random.choice(hours),random.choice(minutes))
            else:
                hour_templates = "{}".format(random.choice(hours)),"{}".format(random.choice(hours))

            templates_to_use = self.TWO_LINE_TIME_TEMPLATES if use_two_line_template else self.ONE_LINE_TIME_TEMPLATES

            line = random.choice(templates_to_use).format(hour_templates[0],hour_templates[1])
            text_lines.append(line)

        hour_text = "\n".join(text_lines)

        use_day_line = self.__random_prob_choice(0.5)

        if use_day_line:
            hour_text += "\n" + self.__generate_day_text()

        return hour_text

    def __generate_sign_text(self):
        return random.choice(self.SIGN_TEMPLATES)
    
    
    def generate_random_text(self):
        use_predefined_text = self.__random_prob_choice(0.3)

        if use_predefined_text:
            return self.__generate_free_sample_text()

        prob = random.uniform(0,1)
        if prob < 0.5:
            return self.__generate_sign_text()

        if prob < 0.75:
            return self.__generate_hour_text()

        return self.__generate_day_text()
    
    
    def generate_texts(self):
        return [self.generate_random_text() for _ in range(self.nr_texts)]
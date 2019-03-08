import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import PIL
import apollo_python_common.io_utils as io_utils
import pandas as pd


class FakeDatasetGenerator:
    TRAIN_PERCENTAGE = 0.99
    IMG_NAME_COL = "img_name"
    TEXT_COL = "text"
    OUTPUT_CSV_NAME = "ocr_data.csv"
    TRAIN_FOLDER = "train"
    TEST_FOLDER = "test"

    def __init__(self, save_path, image_generator, text_generator):
        self.train_save_path = os.path.join(save_path, self.TRAIN_FOLDER)
        self.test_save_path = os.path.join(save_path, self.TEST_FOLDER)
        self.image_generator = image_generator 
        self.text_generator = text_generator
        
    def __split_train_test(self, data, train_percentage):
        nr_train = int(train_percentage * len(data))
        train_data = data[:nr_train]
        test_data = data[nr_train:]

        return train_data, test_data

    def generate_img_and_write_to_disk(self, base_path, img_name_2_text):
        img_name, text = img_name_2_text
        os.makedirs(base_path, exist_ok=True)
        img = self.image_generator.generate_img(text)
    
        PIL.Image.fromarray(img).save(os.path.join(base_path, img_name))

    def __save_csv(self,data,path):
        io_utils.create_folder(path)
        pd.DataFrame(self.preprocess_data(data),columns=[self.IMG_NAME_COL, self.TEXT_COL])\
                    .to_csv(os.path.join(path,self.OUTPUT_CSV_NAME),index=False)
        
    def preprocess_data(self,data):
        return data

    def create_dataset(self):
        texts = self.text_generator.generate_texts()
        img_names_2_texts = [(f"img_{i}.jpg",text) for i,text in enumerate(texts)]

        train_data, test_data = self.__split_train_test(img_names_2_texts, self.TRAIN_PERCENTAGE)
        
        self.__save_csv(train_data,self.train_save_path)
        self.__save_csv(test_data,self.test_save_path)
        
        pool = Pool(multiprocessing.cpu_count() // 2)
        pool.map(partial(self.generate_img_and_write_to_disk, self.train_save_path), train_data)
        pool.close()

        pool = Pool(multiprocessing.cpu_count() // 2)
        pool.map(partial(self.generate_img_and_write_to_disk, self.test_save_path), test_data)
        pool.close()

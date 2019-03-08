import io
import os
from glob import glob

import PIL.Image
import apollo_python_common.io_utils as io_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


class TFRecordDatasetCreator:
    
    IMAGES_PER_FILE = 1000
    MAX_TEXT_LENGTH = 16
    NULL_CHAR_ID = 99
    NUM_OF_VIEWS = 1
    
    IMG_NAME_COL = "img_name"
    TEXT_COL = "text"
    
    def __init__(self, input_path, output_path, charset_path, name_pattern):
        self.input_path = input_path
        self.output_path = output_path
        self.charset = self.__get_charset(charset_path)
        self.name_pattern = name_pattern
        
    def __get_charset(self,charset_path):

        with open(charset_path, encoding="utf-8") as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        charset = {line.split("\t")[1]:int(line.split("\t")[0]) for line in content[1:]}
        charset[' '] = 0
        
        return charset
    
    def __float_feature(self,value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def __int64_feature(self,value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def __bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    def __encode_utf8_string(self,text):
        char_ids_padded = []
        char_ids_unpadded = []

        for letter in text:
            char_ids_padded.append(self.charset[letter])
            char_ids_unpadded.append(self.charset[letter])

        for _ in range(len(text),self.MAX_TEXT_LENGTH):
            char_ids_padded.append(self.NULL_CHAR_ID)

        return char_ids_padded, char_ids_unpadded

    def __compute_example(self,img_path,data_dict):

        text = str(data_dict[os.path.basename(img_path)]).lower()

        with tf.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)

        img = PIL.Image.open(encoded_jpg_io)

        char_ids_padded, char_ids_unpadded = self.__encode_utf8_string(text)

        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image/format': self.__bytes_feature(["jpeg".encode()]),
            'image/encoded': self.__bytes_feature([encoded_jpg]),
            'image/class': self.__int64_feature(char_ids_padded),
            'image/unpadded_class': self.__int64_feature(char_ids_unpadded),
            'height': self.__int64_feature([img.size[1]]),
            'width': self.__int64_feature([img.size[0]]),
            'orig_width': self.__int64_feature([img.size[0]//self.NUM_OF_VIEWS]),
            'image/text': self.__bytes_feature([text.encode()])
          }
        ))

        return example

    def write_split(self,index,img_path_split,data_dict):
        writer = tf.python_io.TFRecordWriter(os.path.join(self.output_path,"{}{}".format(self.name_pattern,str(index))))

        for img_path in img_path_split:
            example = self.__compute_example(img_path,data_dict)
            writer.write(example.SerializeToString())

        writer.close()

    def create_dataset(self):
        io_utils.create_folder(self.output_path)
        
        data_df = pd.read_csv(os.path.join(self.input_path,"ocr_data.csv"))
        data_dict = dict(zip(data_df[self.IMG_NAME_COL],data_df[self.TEXT_COL]))
                         
        img_paths = glob(self.input_path + "/*.jpg")
        print(f"Nr Images = {len(img_paths)}")

        nr_splits = max(1, len(img_paths) // self.IMAGES_PER_FILE)
        print(f"Nr Splits = {nr_splits}")

        img_paths_splits = np.array_split(img_paths,nr_splits)
        for index,img_path_split in tqdm(list(enumerate(img_paths_splits))):
            self.write_split(index,img_path_split,data_dict)
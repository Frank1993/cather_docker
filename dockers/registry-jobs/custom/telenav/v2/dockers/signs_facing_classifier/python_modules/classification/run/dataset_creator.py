import argparse
import logging
import os

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util

import classification.scripts.dataset_builder as builder
import classification.scripts.network as network
from classification.scripts.constants import PipelineParamsBuilder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--dataset_config_json", help="path to json containing dataset params",
                        type=str, required=True)
    return parser.parse_args()


class DatasetCreator:
    
    BASE_PATH_PARAM = "base_path"
    NR_IMAGES_PER_CLASS_PARAM = "nr_images_per_class"
    SUFFIX_PARAM = "suffix"
    WITH_VP_CROP_PARAM = "with_vp_crop"
    IMG_DIM_PARAM = "img_dim"
    SPLIT_TRAIN_TEST_BY_SEQ_ID_PARAM = "split_train_test_by_seq_id"
    AUGMENT_STRATEGIES_PARAM = "augment_strategies"
    SQUARE_IMAGE_PARAM = "square_image"
    KEEP_ASPECT_PARAM = "keep_aspect"
    
    TRAIN_TEST_SPLIT_PERCENTAGE = 0.7
    
    def __init__(self, dataset_config):
        self.dataset_config = dataset_config
        self.params = self.__construct_params()
        self.conv_model = self.__get_backbone_conv_model()
        
    def __serialialize_params(self, params):
        io_utils.json_dump(params, os.path.join(params.params_path, "model_params.json"))
        
    def __construct_params(self):
        params = PipelineParamsBuilder.build_params(self.dataset_config[self.BASE_PATH_PARAM],
                                                  self.dataset_config[self.NR_IMAGES_PER_CLASS_PARAM],
                                                  self.dataset_config[self.SUFFIX_PARAM],
                                                  self.dataset_config[self.WITH_VP_CROP_PARAM],
                                                  self.dataset_config[self.IMG_DIM_PARAM],
                                                  self.dataset_config[self.SQUARE_IMAGE_PARAM],
                                                  self.dataset_config[self.KEEP_ASPECT_PARAM])

        self.__serialialize_params(params)
        return params
        
    def __read_data_from_disk(self):
        return builder.read_data_from_disk(self.params.base_img_path,
                                           self.params.class_2_classIndex,
                                           self.params.nr_images_per_class)

    def train_test_split(self, data_df):
        
        if self.dataset_config[self.SPLIT_TRAIN_TEST_BY_SEQ_ID_PARAM]:
            return builder.split_train_test_by_seq_id(data_df, self.params.train_percentage)
        else:
            return builder.train_test_split(data_df, self.TRAIN_TEST_SPLIT_PERCENTAGE)
        
    def __construct_data_batches(self,df, df_path, img_path):
        builder.construct_data_batches(df,
                                       self.params.img_size,
                                       self.params.base_img_path,
                                       df_path,
                                       img_path,
                                       self.params.nr_entries_per_split,
                                       self.params.with_vp_crop,
                                       self.params.keep_aspect)
        
    def __get_backbone_conv_model(self):
        return network.get_conv_model(self.params.conv_layer_name,self.params.img_size)

    def __augment_train_batches(self):
        augment_dict = self.dataset_config[self.AUGMENT_STRATEGIES_PARAM]
        
        builder.augment_batches(self.params.train_df_path,
                                self.params.train_img_path,
                                self.params.train_conv_path,
                                self.conv_model,
                                augment_dict
                               )
        
        
    def __precompute_conv_on_batches(self,img_path, conv_path):
        builder.precompute_conv_on_batches(img_path, conv_path, self.conv_model)

        
    def __construct_train_test_batches(self):
        
        data_df = self.__read_data_from_disk()
       
        train_data_df, test_data_df = self.train_test_split(data_df)

        self.__construct_data_batches(train_data_df, self.params.train_df_path, self.params.train_img_path)
        self.__construct_data_batches(test_data_df, self.params.test_df_path, self.params.test_img_path) 

     
    def  __precompute_conv_on_train_test_batches(self):
              
        self.__precompute_conv_on_batches(self.params.train_img_path,
                                           self.params.train_conv_path)

        self.__precompute_conv_on_batches(self.params.test_img_path,
                                           self.params.test_conv_path)
        
    def create_dataset(self):
        self.__construct_train_test_batches()
        self.__precompute_conv_on_train_test_batches()
        self.__augment_train_batches()


if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    dataset_config = io_utils.json_load(args.dataset_config_json)    
    
    try:
        DatasetCreator(dataset_config).create_dataset()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

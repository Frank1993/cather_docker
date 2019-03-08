import argparse
import logging
import os
import shutil
from keras.callbacks import ModelCheckpoint

import classification.scripts.network as network
import classification.scripts.generators as generators
import classification.scripts.utils as utils
from classification.scripts.constants import NetworkCfgParams as ncp
import apollo_python_common.ftp_utils as ftp
import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--train_config_json", help="path to json containing training params",
                        type=str, required=True)
    return parser.parse_args()


class NetworkTrainer:
    
    TEMP_PATH = "./model-temp.h5"
    BUNDLE_PATH = "./bundle/"
    WEIGHTS_FILE_NAME = "weights.h5"

    def __init__(self, train_cfg):
        self.logger = logging.getLogger(__name__)
        self.config = train_cfg
        self.train_params = self.__load_params(self.config[ncp.TRAIN_MODEL_PARAMS_PATH])
        if self.config[ncp.WITH_DIFFERENT_TEST_SET_PARAM]:
            self.test_params = self.__load_params(self.config[ncp.TEST_MODEL_PARAMS_PATH])

    def __load_params(self,path):
        return utils.json_load_classif_params(path)
        
    def build_model(self):
        
        network_config = self.config[ncp.NETWORK_CONFIG_PARAM]
        
        input_shape = network.get_conv_model(self.train_params.conv_layer_name, 
                                             self.train_params.img_size).output_shape[1:]
    
        model = network.get_ssd_conv_model(input_shape, 
                                           network_config[ncp.NR_CONV_BLOCKS_PARAM],
                                           network_config[ncp.NR_FILTERS_PARAM],
                                           network_config[ncp.DROPOUT_LEVEL_PARAM],
                                           len(self.train_params.classes))
        
        return model
    
    def get_data_paths(self):
        if self.config[ncp.WITH_DIFFERENT_TEST_SET_PARAM]:
            train_df_path_list = [self.train_params.train_df_path, self.train_params.test_df_path]
            train_img_path_list = [self.train_params.train_img_path, self.train_params.test_img_path]
            train_conv_path_list = [self.train_params.train_conv_path, self.train_params.test_conv_path]

            test_df_path_list = [self.test_params.train_df_path, self.test_params.test_df_path]
            test_img_path_list = [self.test_params.train_img_path, self.test_params.test_img_path]
            test_conv_path_list = [self.test_params.train_conv_path, self.test_params.test_conv_path]

        else:
            train_df_path_list = [self.train_params.train_df_path]
            train_img_path_list = [self.train_params.train_img_path]
            train_conv_path_list = [self.train_params.train_conv_path]

            test_df_path_list = [self.train_params.test_df_path]
            test_img_path_list = [self.train_params.test_img_path]
            test_conv_path_list = [self.train_params.test_conv_path]

        return train_df_path_list, train_img_path_list, train_conv_path_list, \
               test_df_path_list,  test_img_path_list,  test_conv_path_list
    
    def __build_data_generators(self):
       
        train_df_path_list,_,train_conv_path_list,test_df_path_list,_,test_conv_path_list = self.get_data_paths()
        
        train_generator = generators.ClassifGenerator(train_df_path_list, train_conv_path_list)
        test_generator = generators.ClassifGenerator(test_df_path_list, test_conv_path_list)
        
        nr_train_data = sum(len(os.listdir(path)) for path in train_df_path_list)
        nr_test_data = sum(len(os.listdir(path)) for path in test_df_path_list)
        
        return (train_generator, nr_train_data), (test_generator, nr_test_data)
    
    def __train_model(self, model, train_generator_2_count, test_generator_2_count):
        
        train_generator, nr_train_data = train_generator_2_count
        test_generator, nr_test_data = test_generator_2_count

        self.logger.info("Train Batches = {}".format(nr_train_data))
        self.logger.info("Test Batches = {}".format(nr_test_data))

        self.logger.info("Training model...")
        model.fit_generator(train_generator,
                            steps_per_epoch=nr_train_data,
                            epochs=self.config[ncp.NR_EPOCHS_PARAM],
                            validation_data=test_generator,
                            validation_steps=nr_test_data,
                            callbacks=[ModelCheckpoint(self.TEMP_PATH, monitor='val_acc', verbose=1, save_best_only=True,
                                                       mode='max')],
                            workers=10
                            )

        self.logger.info("Loading best weights...")
        model.load_weights(self.TEMP_PATH)
        
        io_utils.create_folder(self.config[ncp.WEIGHTS_LOCAL_OUTPUT_PATH_PARAM])
        model.save_weights(os.path.join(self.config[ncp.WEIGHTS_LOCAL_OUTPUT_PATH_PARAM],self.WEIGHTS_FILE_NAME))
        
        return model
        
    def __save_weights_to_ftp(self,model):
        self.logger.info("Saving to FTP...")
        network.save_model_bundle(self.BUNDLE_PATH, model, self.train_params)

        ftp.dir_copy_local_to_ftp(ftp.FTP_SERVER, ftp.FTP_USER_NAME, ftp.FTP_PASSWORD,
                                  self.BUNDLE_PATH,
                                  self.config[ncp.FTP_OUTPUT_PARAM])
    
    def __clean_workspace(self):
        shutil.rmtree(self.BUNDLE_PATH)
        os.remove(self.TEMP_PATH)
        
    def train_network(self):

        train_generator_2_count, test_generator_2_count = self.__build_data_generators()

        model = self.build_model()
        model = self.__train_model(model, train_generator_2_count, test_generator_2_count)
                
        self.__save_weights_to_ftp(model)  
        self.__clean_workspace()
        

if __name__ == '__main__':

    log_util.config(__file__)

    args = parse_arguments()
    train_config = io_utils.json_load(args.train_config_json)    
    
    try:
        NetworkTrainer(train_config).train_network()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

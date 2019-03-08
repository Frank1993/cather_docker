import argparse
import logging

import numpy as np

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.proto_api as proto_api
from classification.scripts.constants import Column
from classification.scripts.constants import PredictorCfgParams
from classification.scripts.prediction.folder_predictor import FolderPredictor


class Predictor:
    
    def __init__(self, predict_cfg):
        self.predict_config = predict_cfg
        self.folder_predictor = self.__build_folder_predictor()
        
    def __build_folder_predictor(self):
        return FolderPredictor(self.predict_config[PredictorCfgParams.FTP_BUNDLE_PATH_PARAM],
                                                self.predict_config[PredictorCfgParams.NR_IMGS_PARAM],
                                                with_img=False)
        
    def __convert_to_proto(self, pred_df):
        
        classIndex_2_class = self.folder_predictor.params.classIndex_2_class
        
        imageset_proto = proto_api.get_new_metadata_file()
        
        for _, row in pred_df.iterrows():
            img_path = row[Column.FULL_IMG_NAME_COL]
            pred = row[Column.PRED_COL]

            builder_image_proto = proto_api.get_new_image_proto("0", 0, img_path, "", 0, 0)
            image_proto = imageset_proto.images.add()
            image_proto.CopyFrom(builder_image_proto)
            classif_prediction_proto = image_proto.features.classif_predictions.add()
            
            for index, confidence in enumerate(list(pred)):
                classif_prediction_proto_class = classif_prediction_proto.pred_classes.add()
                classif_prediction_proto_class.class_name = classIndex_2_class[index]
                classif_prediction_proto_class.confidence = round(float(confidence), 4)

            classif_prediction_proto.algorithm = ""
            classif_prediction_proto.algorithm_version = ""
            classif_prediction_proto.chosen_pred_class_name = classIndex_2_class[np.argmax(pred)]
            
        return imageset_proto
    
    def __write_proto_to_disk(self,imageset_proto):
        proto_api.serialize_metadata(imageset_proto,self.predict_config[PredictorCfgParams.OUTPUT_FOLDER_PARAM])
            
    def __compute_predictions(self):
        return self.folder_predictor.compute_prediction(self.predict_config[PredictorCfgParams.INPUT_FOLDER_PARAM])
    
    def predict(self):
        pred_df = self.__compute_predictions()
        imageset_proto = self.__convert_to_proto(pred_df)
        self.__write_proto_to_disk(imageset_proto)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--predict_config_json", help="path to json containing predict params",
                        type=str, required=True)
    return parser.parse_args()
    
    
if __name__ == '__main__':

    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    predict_config = io_utils.json_load(args.predict_config_json)    
    
    try:
        Predictor(predict_config).predict()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

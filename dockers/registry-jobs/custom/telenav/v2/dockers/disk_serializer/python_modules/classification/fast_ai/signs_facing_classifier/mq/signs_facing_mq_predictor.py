import argparse
import logging
import os

from fastai.vision import *

import classification.fast_ai.signs_facing_classifier.utils.constants as const
from apollo_python_common import image, proto_api, io_utils, log_util
from apollo_python_common.ml_pipeline.multi_threaded_predictor import MultiThreadedPredictor
from classification.fast_ai.common import predict
from classification.fast_ai.signs_facing_classifier.utils.constants import SignFacingLabel


class SignsFacingMQPredictor(MultiThreadedPredictor):
    LABEL_VALUE_DICT = {SignFacingLabel.FRONT: 0, SignFacingLabel.LEFT: -1, SignFacingLabel.RIGHT: 1}

    ES_KEY_ROI_FACING = "roi_facing_prediction"
    # this is a default invalid value for an angle, since the local protobuf object has a required fields that won't be
    # filled by this component.
    BOGUS_ANGLE_VALUE = -400

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.classes = io_utils.json_load(os.path.join(self.config.model_dir, self.config.label_list_file)).label_list
        logger.info("labels: {}".format(self.classes))

        self.learner = predict.get_inference_learner(self.config, self.classes,
                                                     const.MODEL_DICT[self.config.backbone_model])

    def _read_full_img(self, image_proto):
        """ Reads the full image, given an image proto. """

        osc_details = image.OscDetails(image_proto.metadata.id, self.config.osc_api_url)
        img = image.get_rgb(image_proto.metadata.image_path, osc_details)
        proto_api.add_image_size(image_proto, img.shape)

        return img

    def _predict_batch(self, rois_2_ids):
        if len(rois_2_ids) == 0:
            return []

        rois, ids = zip(*rois_2_ids)
        preds = [self.learner.predict(roi) for roi in rois]

        predictions_2_ids = list(zip(preds, ids))

        return predictions_2_ids

    def _update_angle_of_roi_inplace(self, roi_proto, value):
        """ Updates the roi proto, setting a value in the local field for the angle_of_roi. This value is not an actual
         angle but can be -1 for left oriented signs, 1 for right oriented signs, and 0 for front oriented signs. """

        logger.info('updating angle of roi_id {} - value {} '.format(predict.roi_2_id(roi_proto), value))

        proto_local = proto_api.get_new_processed_sign_localization_proto(self.BOGUS_ANGLE_VALUE,
                                                                          self.BOGUS_ANGLE_VALUE, 0,
                                                                          self.BOGUS_ANGLE_VALUE,
                                                                          self.BOGUS_ANGLE_VALUE, value)
        roi_proto.local.CopyFrom(proto_local)

    def _get_pred_value(self, prediction):
        """ Returns the value to be saved in angle_of_roi, based on a given prediction.
         This angle_of_roi value is an integer of -1 for left orientation, 0 for front and 1 for right. This way, we can
         update with a real angle value based on our tests (this may be 90 degrees, or we may decide that 60 degrees is a
          better value). """

        pred_class = str(prediction[0])
        logger.info("prediction class: {}".format(pred_class))
        value = self.LABEL_VALUE_DICT[pred_class]

        logger.info('the resulting angle of roi value: {}'.format(value))
        return value

    def _update_results_in_proto(self, predictions_2_ids, image_proto):
        """ If the rois in the image_proto are available in predictions_2_ids it means we have a prediction for
         the respective roi and we will apply best threhsolds on that. If the roi isn't available we automatically set
         the result class to 'front', meaning the angle of roi will be 0. """

        pred_dict = {roi_id: pred for pred, roi_id in predictions_2_ids}
        image_rois = image_proto.rois

        for roi_proto in image_rois:
            roi_id = predict.roi_2_id(roi_proto)
            self._update_angle_of_roi_inplace(roi_proto, self._get_pred_value(pred_dict[roi_id]))

            logger.info(
                'update results in proto - we have a prediction {} for roi_id {}'.format(pred_dict[roi_id], roi_id))

    def _get_class_name_with_confidence(self, pred, index):
        return "{}:{}".format(self.classes[index], round(float(pred[2][index]), 4))

    def _audit_predictions(self, image_proto, predictions_2_ids):

        pred_2_ids_str = ["{},{},{},{}".format(roi_id,
                                            self._get_class_name_with_confidence(pred, 0),
                                            self._get_class_name_with_confidence(pred, 1),
                                            self._get_class_name_with_confidence(pred, 2))
                          for pred, roi_id in predictions_2_ids]

        super().set_audit_key_val(image_proto.metadata.image_path, self.ES_KEY_ROI_FACING, pred_2_ids_str)

    def preprocess(self, image_proto):
        if image_proto.rois is None or len(image_proto.rois) == 0:
            return list()

        img = self._read_full_img(image_proto)
        rois_2_ids = [
            predict.extract_cropped_roi(img, roi_proto, self.config.image_size, self.config.sq_crop_factor) for
            roi_proto in image_proto.rois]

        return rois_2_ids

    def predict(self, rois_2_ids_lists):
        if self.learner is None:
            self.learner = predict.get_inference_learner(self.config, self.classes,
                                                         const.MODEL_DICT[self.config.backbone_model])

        predictions_2_ids_list = [self._predict_batch(rois_2_ids) for rois_2_ids in rois_2_ids_lists]

        logger.debug("predict - predictions: {}".format(predictions_2_ids_list))

        return predictions_2_ids_list

    def postprocess(self, predictions_2_ids, image_proto):
        self._update_results_in_proto(predictions_2_ids, image_proto)
        self._audit_predictions(image_proto, predictions_2_ids)

        logger.info("resulting proto {}".format(image_proto))

        return image_proto


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", help="config file path", type=str, required=True)

    return parser.parse_args()


def run_predictor(conf_file):
    conf = io_utils.config_load(conf_file)
    predictor = SignsFacingMQPredictor(conf)
    predictor.start()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()
    run_predictor(args.config_file)

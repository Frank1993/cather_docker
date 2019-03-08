import os

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import numpy as np

import apollo_python_common.proto_api as proto_api


class RoiPredictionsPostprocessor:
    PRED_COL = "pred"
    SHORT_IMG_NAME_COL = "short_img_name"
    IMG_NAME_COL = "img_name"
    IS_FP_COL = "is_fp"
    OUTPUT_PREFIX_NAME = "postprocessed_preds"

    def __init__(self, initial_roi_path, classif_pred_path, output_folder):
        self.initial_roi_path = initial_roi_path
        self.classif_pred_path = classif_pred_path
        self.output_folder = os.path.join(output_folder, self.OUTPUT_PREFIX_NAME + "_{}")

    def __is_false_positive(self, row, conf_threshold):
        bad_conf = row[self.PRED_COL][0]
        return bad_conf > conf_threshold

    def __full_2_short_img_name(self, s):
        return "_".join(s.split("_")[2:])

    def __df_2_dict(self, data_df, conf_threshold):
        data_df = data_df[[self.PRED_COL]].reset_index()
        data_df[self.SHORT_IMG_NAME_COL] = data_df[self.IMG_NAME_COL].apply(self.__full_2_short_img_name)
        data_df[self.IS_FP_COL] = data_df.apply(lambda r: self.__is_false_positive(r, conf_threshold), axis=1)
        data_df = data_df.drop([self.IMG_NAME_COL, self.PRED_COL], axis=1)
        data_df = data_df.set_index(self.SHORT_IMG_NAME_COL)

        d = {i: row[self.IS_FP_COL] for i, row in data_df.iterrows()}

        return d

    def __get_roi_id(self, image_proto, roi_proto):
        return "{}_{}-{}-{}-{}.jpg".format(os.path.basename(image_proto.metadata.image_path).split(".")[0],
                                           roi_proto.rect.tl.col,
                                           roi_proto.rect.tl.row,
                                           roi_proto.rect.br.col,
                                           roi_proto.rect.br.row)

    def __write_postprocessed_roi_with_remove(self, conf_threshold):

        classif_pred_df = pd.read_pickle(self.classif_pred_path)
        fp_dict = self.__df_2_dict(classif_pred_df, conf_threshold)

        imageset_proto = proto_api.read_imageset_file(self.initial_roi_path)
        image_proto_list = imageset_proto.images

        nr_changed_rois = 0
        nr_total_rois = 0

        for image_proto in tqdm(image_proto_list):
            rois_to_delete = []
            nr_total_rois += len(image_proto.rois)

            for roi_proto in image_proto.rois:
                key = self.__get_roi_id(image_proto, roi_proto)

                if key not in fp_dict:
                    continue

                if fp_dict[key]:
                    rois_to_delete.append(roi_proto)
                    nr_changed_rois += 1

            for roi in rois_to_delete:
                image_proto.rois.remove(roi)

        print("Changed {} out of {} => {}".format(nr_changed_rois, nr_total_rois,
                                                  round(nr_changed_rois / nr_total_rois, 3)))

        output_path = self.output_folder.format(conf_threshold)
        proto_api.serialize_proto_instance(imageset_proto, output_path)

    def postprocess_predictions(self, conf_thresholds=None):
        if conf_thresholds is None:
            conf_thresholds = [round(x, 3) for x in np.arange(0.3, 1.0, 0.05)]

        for conf_threshold in tqdm(conf_thresholds):
            self.__write_postprocessed_roi_with_remove(conf_threshold)

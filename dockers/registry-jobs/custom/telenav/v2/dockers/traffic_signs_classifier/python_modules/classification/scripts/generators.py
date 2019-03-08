import numpy as np
import pandas as pd
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc_v3
from keras.utils import Sequence

import classification.scripts.dataset_builder as builder
import classification.scripts.utils as utils
from classification.scripts.constants import Column


class ClassifGenerator(Sequence):

    def __init__(self, df_path_list, conv_path_list):
        self.batch_df_paths = []
        self.batch_conv_paths = []

        for df_path, conv_path in zip(df_path_list, conv_path_list):
            split_batch_df_paths = utils.read_file_paths(df_path)
            split_batch_conv_paths = utils.read_file_paths(conv_path)

            self.batch_df_paths += split_batch_df_paths
            self.batch_conv_paths += split_batch_conv_paths

    def __len__(self):
        return len(self.batch_df_paths)

    def __getitem__(self, idx):
        batch_df = pd.read_pickle(self.batch_df_paths[idx])
        conv_img = np.load(self.batch_conv_paths[idx])
        label_classes = utils.numpify(batch_df[Column.LABEL_COL])

        return conv_img, label_classes


class ClassifInferenceGenerator(Sequence):

    def __init__(self, pred_df_splits, img_size, with_vp_crop, with_img, keep_aspect=False):
        self.pred_df_splits = pred_df_splits
        self.img_size = img_size
        self.with_vp_crop = with_vp_crop
        self.with_img = with_img
        self.keep_aspect = keep_aspect

        self.pred_df_dict = {}
        self.filtered_out_df_dict = {}

    def __len__(self):
        return len(self.pred_df_splits)

    def get_data(self):
        pred_df_list = sorted(list(self.pred_df_dict.items()), key=lambda t: t[0])
        filtered_out_df_list = sorted(list(self.filtered_out_df_dict.items()), key=lambda t: t[0])

        pred_df_list = [v for _, v in pred_df_list]
        filtered_out_df_list = [v for _, v in filtered_out_df_list]

        return pred_df_list, filtered_out_df_list

    def __getitem__(self, idx):
        pred_df = self.pred_df_splits[idx]
        pred_df = pd.concat([pred_df, pred_df.loc[:, Column.FULL_IMG_NAME_COL].apply(lambda full_img_name
                                                                                     : builder.read_image(full_img_name,
                                                                                                          self.img_size,
                                                                                                          self.with_vp_crop,
                                                                                                          self.keep_aspect))],
                            axis=1)

        pred_df = builder.add_vp_height_data(pred_df, self.with_vp_crop)

        filtered_out_df = pred_df[~pred_df[Column.VALID_HEIGHT_RATIO_COL]]
        pred_df = pred_df[pred_df[Column.VALID_HEIGHT_RATIO_COL]]

        img_data = utils.numpify(pred_df[Column.IMG_COL]).astype(np.float32)
        img_data = preprocess_input_inc_v3(img_data)

        if not self.with_img:
            pred_df = pred_df.drop([Column.IMG_COL], axis=1)
            filtered_out_df = filtered_out_df.drop([Column.IMG_COL], axis=1)

        self.pred_df_dict[idx] = pred_df
        self.filtered_out_df_dict[idx] = filtered_out_df

        return img_data
import multiprocessing
import os

from tqdm import tqdm

tqdm.pandas()
import numpy as np
import pandas as pd

import classification.scripts.utils as utils
from classification.scripts.constants import Column, INVALID_IMG_PRED
import classification.scripts.network as network
import classification.scripts.generators as generators


class FolderPredictor():

    def __init__(self, ftp_bundle_path, nr_imgs=None, with_img=False, nr_workers=None, tf_version=False):
        self.nr_imgs = nr_imgs
        self.with_img = with_img
        self.nr_workers = nr_workers if nr_workers is not None else multiprocessing.cpu_count() // 2
        self.nr_entries_per_split = 32
        self.tf_version = tf_version

        self.model, self.params = network.load_model_bundle_from_ftp(ftp_bundle_path, tf_version)

    def tf_predict(self, infer_generator):
        session = self.model

        input_tensor = session.graph.get_tensor_by_name("classif/model_1_input:0")
        output_tensor = session.graph.get_tensor_by_name("classif/model_2/dense_5/Softmax:0")

        all_preds = []

        for index in tqdm(range(infer_generator.__len__())):
            imgs = infer_generator.__getitem__(index)
            preds = session.run([output_tensor], feed_dict={input_tensor: imgs})
            preds = np.squeeze(np.asarray(preds))
            all_preds.append(preds)

        return np.vstack(all_preds)

    def keras_predict(self, infer_generator):
        return self.model.predict_generator(infer_generator,
                                            verbose=1,
                                            workers=self.nr_workers)

    def get_predictions(self, infer_generator):
        return self.tf_predict(infer_generator) if self.tf_version else self.keras_predict(infer_generator)

    def read_paths(self, pred_img_path):
        pred_img_list = utils.read_image_paths(pred_img_path)

        if self.nr_imgs is not None:
            pred_img_list = pred_img_list[:self.nr_imgs]

        pred_df = pd.DataFrame({
            Column.IMG_NAME_COL: [os.path.basename(path) for path in pred_img_list],
            Column.FULL_IMG_NAME_COL: pred_img_list,
            Column.WAY_ID_COL: [os.path.basename(path).split("_")[0] for path in pred_img_list],
        })

        pred_df = pred_df.set_index([Column.IMG_NAME_COL])

        nr_splits = max(1, int(float(len(pred_df)) / self.nr_entries_per_split) + 1)

        pred_df_splits = np.array_split(pred_df, nr_splits)

        return pred_df_splits

    def compute_prediction(self, pred_img_path):
        pred_df_splits = self.read_paths(pred_img_path)
        infer_generator = generators.ClassifInferenceGenerator(pred_df_splits,
                                                               self.params.img_size,
                                                               self.params.with_vp_crop,
                                                               self.with_img,
                                                               self.params.keep_aspect)

        predictions = self.get_predictions(infer_generator)

        pred_df_list, filtered_out_df_list = infer_generator.get_data()
        pred_df = pd.concat(pred_df_list)
        filtered_out_df = pd.concat(filtered_out_df_list)

        prediction_df = self.merge_predictions_with_df(pred_df, filtered_out_df, predictions)
        prediction_df.loc[:, Column.PRED_CONF_COL] = prediction_df.loc[:, Column.PRED_COL].apply(lambda pred: max(pred))

        return prediction_df

    def merge_predictions_with_df(self, pred_df, filtered_out_df, predictions):
        pred_df.loc[:, Column.PRED_COL] = pd.Series([pred for pred in predictions], index=pred_df.index)
        pred_df.loc[:, Column.PRED_CLASS_COL] = pred_df.loc[:, Column.PRED_COL].apply(
            lambda pred: utils.label2text(pred, self.params.classIndex_2_class))

        filtered_out_df[Column.PRED_COL] = [np.zeros(len(self.params.classIndex_2_class)) for _ in
                                            range(len(filtered_out_df))]
        filtered_out_df[Column.PRED_CLASS_COL] = INVALID_IMG_PRED

        concat_df = pd.concat([pred_df, filtered_out_df])

        return concat_df

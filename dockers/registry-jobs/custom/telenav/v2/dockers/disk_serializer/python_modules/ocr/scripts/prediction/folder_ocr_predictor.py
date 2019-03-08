import os
from glob import glob

import apollo_python_common.image as image_api
import editdistance
import numpy as np
import pandas as pd
from ocr.scripts.prediction.ocr_predictor import OCRPredictor, ResizeType
from tqdm import tqdm


class FolderOCRPredictor(OCRPredictor):
    IMG_PATH = "img_path"
    IMG = "img"
    RESIZED_IMG = "resized_img"
    PRED_TEXT = "pred_text"
    PRED_CONF = "pred_conf"
    GT_TEXT = "gt_text"
    WIDTH = "width"
    HEIGHT = "height"
    AREA = "area"
    ERROR = "error"
    ERROR_BUCKET = "error_bucket"
    HEIGHT_BUCKET = "height_bucket"
    WIDTH_BUCKET = "width_bucket"
    AREA_BUCKET = "area_bucket"
    CORRECT = "correct"
    ASPECT_RATIO = "aspect_ratio"
    GT_CHAR_LENGTH = "gt_char_length"
    GT_WORD_LENGTH = "gt_word_length"

    def __init__(self, dataset_name, ckpt_path, resize_type, text_corrector=None):
        super().__init__(dataset_name, ckpt_path, text_corrector)
        self.resize_type = resize_type

    def __compute_pred_df(self, pred_df, pred_2_confs, with_gt):
        pred_df[self.PRED_TEXT] = pd.Series([p for p, c in pred_2_confs], index=pred_df.index)
        pred_df[self.PRED_TEXT] = pred_df[self.PRED_TEXT].progress_apply(self.postprocess_text)
        pred_df[self.PRED_CONF] = pd.Series([c for p, c in pred_2_confs], index=pred_df.index)

        if with_gt:
            pred_df[self.GT_TEXT] = pred_df[self.IMG_PATH].apply(lambda img_path: self.__get_gt_from_img_path(img_path))
            pred_df[self.ERROR] = pred_df.apply(lambda r: self.__compute_error(r[self.GT_TEXT], r[self.PRED_TEXT]),
                                                axis=1)
            pred_df[self.CORRECT] = pred_df[self.ERROR].apply(lambda er: 1 if er == 0 else 0)
            pred_df[self.GT_CHAR_LENGTH] = pred_df[self.GT_TEXT].apply(len)
            pred_df[self.GT_WORD_LENGTH] = pred_df[self.GT_TEXT].apply(lambda pred: len(pred.split(" ")))

        return pred_df

    def __compute_error(self, gt, pred):
        return editdistance.eval(gt, pred)

    def __get_gt_from_img_path(self, img_path):
        return os.path.basename(os.path.splitext(img_path)[0]).split("_")[-1].lower().replace(":", "/")

    def __get_data_df(self, img_paths, nr_imgs, min_component_size):
        data_df = pd.DataFrame({self.IMG_PATH: img_paths})
        data_df[self.IMG] = data_df[self.IMG_PATH].apply(image_api.get_rgb)
        data_df[self.RESIZED_IMG] = data_df[self.IMG].apply(self.__resize_img)
        data_df[self.WIDTH] = data_df[self.IMG].apply(lambda img: img.shape[1])
        data_df[self.HEIGHT] = data_df[self.IMG].apply(lambda img: img.shape[0])

        data_df[self.AREA] = data_df[self.WIDTH] * data_df[self.HEIGHT]
        data_df[self.ASPECT_RATIO] = data_df.apply(lambda row: row[self.WIDTH] // row[self.HEIGHT], axis=1)

        data_df = data_df[data_df[self.HEIGHT] > min_component_size]

        data_df = data_df[:nr_imgs]

        return data_df

    def __resize_img(self, img):
        if self.resize_type == ResizeType.PAD:
            resized_img, _, _ = image_api.resize_image_fill(img, self.height, self.width, 3)
            return resized_img

        if self.resize_type == ResizeType.DEFORM:
            return image_api.cv_resize(img, self.width, self.height, default_interpolation=False)

        print("Invalid ResizeType. No resize performed. Must be 'deform' or 'pad'")
        return img

    def __get_imgs(self, data_df):
        return np.stack(data_df[self.RESIZED_IMG].values)

    def predict_on_folder(self, predict_folder, nr_imgs=None, with_gt=False, min_component_size=0):
        img_paths = glob(predict_folder + "/*.jp*")
        nr_imgs = len(img_paths) if nr_imgs is None else nr_imgs
        data_df = self.__get_data_df(img_paths, nr_imgs, min_component_size)
        imgs = self.__get_imgs(data_df)

        pred_2_conf = [self.make_prediction_on_img(img) for img in tqdm(imgs)]
        pred_df = self.__compute_pred_df(data_df, pred_2_conf, with_gt)

        return pred_df

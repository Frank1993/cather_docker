import logging
import logging.config
import multiprocessing
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from PIL import ImageFile
from keras.applications.inception_v3 import preprocess_input as preprocess_input_inc_v3
from sklearn.utils import shuffle
from tqdm import tqdm

import apollo_python_common.image
import apollo_python_common.image as image
import apollo_python_common.io_utils as io_utils
import classification.scripts.utils as utils
from classification.scripts.constants import Column
from vanishing_point.vanishing_point import VanishingPointDetector

tqdm.pandas()
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BatchParams:
    def __init__(self, base_path, df_save_folder, img_save_folder, index, image_size, with_vp_crop, keep_aspect):
        '''
        :param base_path: base path for the work directory
        :param df_save_folder: folder where the dataframe batch is located
        :param img_save_folder: folder where the image data is saved for the batch
        :param index: the index of the batch after the split
        :param image_size: size of the images in the batch
        :param with_vp_crop: determines if the image is cut at Vanishing Point or not
        :param keep_aspect: determines if the image is resized keeping it's aspect ratio or, not.
        '''
        self.base_path = base_path
        self.df_save_folder = df_save_folder
        self.img_save_folder = img_save_folder
        self.index = index
        self.image_size = image_size
        self.with_vp_crop = with_vp_crop
        self.keep_aspect = keep_aspect


class BatchData:
    def __init__(self, batch_data, batch_params):
        '''
        :param batch_data: the actual batch data
        :param batch_params: parameters specific to the creation of the batch
        '''
        self.batch_data = batch_data
        self.batch_params = batch_params


def split_train_test_by_seq_id(data_df, train_percentage):
    seq_ids = list(set(data_df[Column.SEQ_ID_COL]))

    nr_train_ids = max(1, int(len(seq_ids) * train_percentage))

    train_seq_ids = seq_ids[:nr_train_ids]
    test_seq_ids = seq_ids[nr_train_ids:]

    train_data_df = data_df.loc[data_df[Column.SEQ_ID_COL].isin(train_seq_ids)].sample(frac=1)
    test_data_df = data_df.loc[data_df[Column.SEQ_ID_COL].isin(test_seq_ids)].sample(frac=1)

    logger = logging.getLogger(__name__)
    logger.info("All Count = {}".format(len(data_df)))
    logger.info("Train Count = {}".format(len(train_data_df)))
    logger.info("Test Count = {}".format(len(test_data_df)))

    return train_data_df, test_data_df


def train_test_split(data_df, train_percentage):
    nr_train = int(train_percentage * len(data_df))
    data_df = shuffle(data_df)
    train_df = data_df[:nr_train]
    test_df = data_df[nr_train:]
    
    logger = logging.getLogger(__name__)
    logger.info("All Count = {}".format(len(data_df)))
    logger.info("Train Count = {}".format(len(train_df)))
    logger.info("Test Count = {}".format(len(test_df)))
    
    return train_df, test_df


def read_data_from_disk(base_img_path, class_2_classIndex, nr_images_per_class=None, with_label=True):
    logger = logging.getLogger(__name__)
    logger.info("Reading data from disk...")

    class_df_list = []

    for current_class in sorted(class_2_classIndex.keys()):

        class_path = os.path.join(base_img_path,current_class)

        img_names = sorted(os.listdir(class_path))

        if nr_images_per_class is not None:
            img_names = img_names[:nr_images_per_class]

        class_df = pd.DataFrame({
            Column.IMG_NAME_COL: img_names,
            Column.WAY_ID_COL: [name.split("_")[0] for name in img_names],
            Column.SEQ_ID_COL: [name.split("_")[1] for name in img_names]
        })

        class_df.loc[:, Column.LABEL_CLASS_COL] = current_class

        class_df_list.append(class_df)

    data_df = pd.concat(class_df_list, ignore_index=True)

    if with_label:
        labels_ohe = utils.transform_to_ohe(data_df[Column.LABEL_CLASS_COL].tolist(), class_2_classIndex)
        data_df.loc[:, Column.LABEL_COL] = pd.Series([label for label in labels_ohe])
    else:
        data_df.loc[:, Column.LABEL_COL] = np.asarray([-1])

    data_df = shuffle(data_df, random_state=0)

    return data_df

def mock_image_entry(new_img_size):
    mock_image = np.zeros((new_img_size[0], new_img_size[1], 3)).astype(np.uint8)
    return pd.Series({Column.IMG_COL: mock_image,
                      Column.HEIGHT_RATIO_COL: 1,
                      Column.HEIGHT_BEFORE_RESIZE_COL: 0})


def valid_vp(vp, img_height):
    return vp is not None and 0 < vp.y < img_height


def crop_at_horizon_line(img):
    original_img_height, original_img_width, _ = img.shape

    vanishing_point, _ = VanishingPointDetector().get_vanishing_point(img)

    if valid_vp(vanishing_point, original_img_height):
        img = img[vanishing_point.y:original_img_height, 0:original_img_width]

    return img


def read_image(img_path, new_img_size, with_vp_crop, keep_aspect=False):
    logger = logging.getLogger(__name__)
    try:
        img = image.get_rgb(img_path)
    except Exception as e:
        logger.error("Recovered from : {}".format(e), exc_info=True)
        return mock_image_entry(new_img_size)

    height, _, _ = img.shape

    if with_vp_crop:
        img = crop_at_horizon_line(img)

    height_before_resize, _, _ = img.shape
    if keep_aspect:
        img, _, _ = image.resize_image_fill(img, new_img_size[1], new_img_size[0], 3)
    else:
        img = image.cv_resize(img, new_img_size[0], new_img_size[1])

    return pd.Series({Column.IMG_COL: img,
                      Column.HEIGHT_RATIO_COL: height_before_resize / height,
                      Column.HEIGHT_BEFORE_RESIZE_COL: height_before_resize})


def add_vp_height_data(data_df, with_vp_crop):
    hr_min_level = 0.15
    hr_max_level = 0.65

    data_df.loc[:, Column.VALID_HEIGHT_RATIO_COL] = data_df.loc[:, Column.HEIGHT_RATIO_COL].apply(
        lambda hr: hr_min_level < hr < hr_max_level) if with_vp_crop else True

    return data_df


def construct_batch(df_batch):
    logger = logging.getLogger(__name__)

    data_df_split = df_batch.batch_data
    batch_params = df_batch.batch_params

    data_df_split.loc[:, Column.FULL_IMG_NAME_COL] = data_df_split.apply(lambda row: os.path.join(batch_params.base_path,
                                                                                                  row[Column.LABEL_CLASS_COL],
                                                                                                  row[Column.IMG_NAME_COL]),
                                                                         axis=1)

    data_df_split = data_df_split.set_index([Column.IMG_NAME_COL])

    data_df_split = pd.concat([data_df_split,
                               data_df_split.loc[:, Column.FULL_IMG_NAME_COL].apply(lambda full_img_name:
                                                                                    read_image(full_img_name,
                                                                                               batch_params.image_size,
                                                                                               batch_params.with_vp_crop,
                                                                                               batch_params.keep_aspect)
                                                                                    )
                               ],
                              axis=1)

    data_df_split = add_vp_height_data(data_df_split, batch_params.with_vp_crop)
    data_df_split = data_df_split[data_df_split[Column.VALID_HEIGHT_RATIO_COL]]

    if len(data_df_split) == 0:
        logger.info("Batch is empty...")
        return

    img_data = utils.numpify(data_df_split[Column.IMG_COL])

    data_df_split = data_df_split.drop(Column.IMG_COL, axis=1).drop(Column.FULL_IMG_NAME_COL, axis=1)

    batch_save_name = "data_df_split_{}".format(format(batch_params.index, "04"))

    df_batch_save_name = "{}_df.pkl".format(batch_save_name)
    img_batch_save_name = "{}_img.dat".format(batch_save_name)
    
    data_df_split.to_pickle(os.path.join(batch_params.df_save_folder,df_batch_save_name))

    np.save(os.path.join(batch_params.img_save_folder,img_batch_save_name), img_data)


def construct_data_batches(data_df,
                           new_img_size,
                           base_img_path,
                           df_save_folder,
                           img_save_folder,
                           nr_entries_per_split,
                           with_vp_crop,
                           keep_aspect=False):
    logger = logging.getLogger(__name__)

    logger.info("Constructing batches...")

    io_utils.create_folder(df_save_folder)
    io_utils.create_folder(img_save_folder)

    nr_splits = max(1, int(float(len(data_df)) / nr_entries_per_split) + 1)

    logger.info("Number of splits = {}".format(nr_splits))

    data_df_splits = np.array_split(data_df, nr_splits)

    all_data = []
    for index, data_df_split in enumerate(data_df_splits):
        all_data.append(BatchData(data_df_split,
                                  BatchParams(base_img_path,
                                              df_save_folder,
                                              img_save_folder,
                                              index,
                                              new_img_size,
                                              with_vp_crop,
                                              keep_aspect)))

    threads_number = multiprocessing.cpu_count() // 2

    pool = Pool(threads_number)
    pool.map(construct_batch, all_data)
    pool.close()


def precompute_conv_on_batches(input_folder, output_folder, conv_model, index_to_start_from=0):
    logger = logging.getLogger(__name__)
    logger.info("Precomputing conv output...")

    io_utils.create_folder(output_folder)

    batch_paths = utils.read_file_names(input_folder)

    logger.info("Number of batches = {}".format(len(batch_paths)))

    for index, batch_path in tqdm(list(enumerate(batch_paths))):

        if index < index_to_start_from:
            logger.info("Skipping: {}".format(batch_path))
            continue

        batch_name = batch_path.split(".")[0]

        img_data = np.load(os.path.join(input_folder,batch_path)).astype(np.float32)

        img_data = preprocess_input_inc_v3(img_data)

        conv_img_data = conv_model.predict(img_data, batch_size=48, verbose=1)

        np.save(os.path.join(output_folder,"{}_conv.dat".format(batch_name)), conv_img_data)


def rotate_image_cw(img):
    return apollo_python_common.image.rotate_image(img, -15)


def rotate_image_ccw(img):
    return apollo_python_common.image.rotate_image(img, 15)


def get_augment_name_2_function_dict():
    return {
        "flip": apollo_python_common.image.flip_image,
        "rotate_cw": rotate_image_cw,
        "rotate_ccw": rotate_image_ccw
    }


def get_augment_types(augment_dict):
    return [augment_type for augment_type, should_augment in augment_dict.items() if should_augment]


def augment_batches(df_path, img_path, conv_path, conv_model, augment_dict, start_index=0):
    augment_types = get_augment_types(augment_dict)
    logger = logging.getLogger(__name__)

    if len(augment_types) == 0:
        logger.info("No augmentation applied")
        return

    logger.info("Augmenting batches...")

    batch_paths = sorted([batch_path for batch_path in os.listdir(df_path) if "augm" not in batch_path])

    for index, batch_path in tqdm(list(enumerate(batch_paths))):

        if index < start_index:
            logger.info("Skipping: {}".format(batch_path))
            continue

        batch_name = batch_path.split(".")[0][:-3]

        batch_data_df = pd.read_pickle(os.path.join(df_path,batch_path))
        img_data = np.load(os.path.join(img_path,"{}_img.dat.npy".format(batch_name)))

        augment_name_2_function_dict = get_augment_name_2_function_dict()

        for augment_type in augment_types:
            augment_function = augment_name_2_function_dict[augment_type]

            augm_df_batch_name = "{}_{}_df_augm".format(batch_name,augment_type)
            augm_conv_batch_name = "{}_{}_conv_augm".format(batch_name,augment_type)
            augm_img_batch_name = "{}_{}_img_augm".format(batch_name,augment_type)

            augm_img_data = np.stack([augment_function(img) for img in img_data])

            augm_img_data_format = preprocess_input_inc_v3(augm_img_data.astype(np.float32))

            augm_conv_img_data = conv_model.predict(augm_img_data_format, verbose=1)

            batch_data_df.to_pickle(os.path.join(df_path, "{}.pkl".format(augm_df_batch_name)))  # same df
            np.save(os.path.join(img_path, "{}.dat".format(augm_img_batch_name)), augm_img_data)
            np.save(os.path.join(conv_path, "{}.dat".format(augm_conv_batch_name)), augm_conv_img_data)

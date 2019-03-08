import os
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd

import apollo_python_common.io_utils as io_utils
from apollo_python_common.io_utils import IMG_EXTENSIONS
from keras.utils.np_utils import to_categorical


def label2text(label, classIndex_2_class):
    return classIndex_2_class[np.argmax(label)]


def onehot(x):
    return to_categorical(x)


def transform_to_ohe(label_data, class_2_classIndex):
    label_data_indexed = [class_2_classIndex[label] for label in label_data]
    label_data_ohe = onehot(np.asarray(label_data_indexed))
    return label_data_ohe


def numpify(df_series):
    return np.stack(df_series.tolist())


def merge_df_with_data(df_path, data_path, data_col_name):
    batch_df = pd.read_pickle(df_path)

    data = np.load(data_path)
    
    batch_df.loc[:, data_col_name] = pd.Series([d for d in data],index=batch_df.index)

    return batch_df


def get_nr_of_instances(df_path):
    split_paths = read_file_names(df_path)

    nr_instances = 0

    for split_path in tqdm(split_paths):
        data_df_split = pd.read_pickle(df_path + split_path)

        nr_instances += len(data_df_split)

    return nr_instances


def read_data_batches_in_df(df_path, data_path, data_col_name, nr_batches=None):
    df_batches_paths = read_file_paths(df_path)
    data_batches_paths = read_file_paths(data_path)

    if nr_batches is not None:
        df_batches_paths = df_batches_paths[:nr_batches]
        data_batches_paths = data_batches_paths[:nr_batches]

    final_data_df = pd.concat([merge_df_with_data(df_path, data_path, data_col_name) \
                               for (df_path, data_path) in tqdm(zip(df_batches_paths, data_batches_paths))], axis=0)

    return final_data_df


def read_img_batches_in_df(df_path, data_path, nr_batches=None):
    return read_data_batches_in_df(df_path, data_path, 'img', nr_batches)


def read_conv_batches_in_df(df_path, data_path, nr_batches=None):
    return read_data_batches_in_df(df_path, data_path, 'conv_img', nr_batches)


def shorten(df):
    collist = [col for col in df.columns if col not in ['img', 'conv_img']]
    return df[collist]


def get_way_id_preds_df(pred_df):
    wayId_2_indexList = pred_df.reset_index() \
        .groupby("way_id")['img_name'] \
        .apply(list) \
        .to_dict()

    way_id_preds = []
    way_ids = []
    way_id_ground_truth = []

    for way_id, index_list in wayId_2_indexList.items():
        way_id_predictions = pred_df['pred'].ix[index_list]
        voting_prediction = np.average(way_id_predictions, axis=0)

        way_id_label = pred_df['label'].ix[index_list].tolist()[0]  # make sure they are the same

        way_id_preds.append(voting_prediction)
        way_id_ground_truth.append(way_id_label)
        way_ids.append(way_id)

    way_id_pred_df = pd.DataFrame(
        {'way_id': way_ids,
         'pred': way_id_preds,
         'label': way_id_ground_truth
         }
    )

    way_id_pred_df.loc[:, 'correct'] = way_id_pred_df.apply(
        lambda row: 1 if np.argmax(row['pred']) == np.argmax(row['label']) else 0, axis=1)

    return way_id_pred_df


def merge_dfs(img_data_df, conv_data_df):
    conv_data_df = conv_data_df[['conv_img']]
    return pd.concat([img_data_df, conv_data_df], axis=1)


def read_image_paths(path):
    img_paths = []
    for ext in IMG_EXTENSIONS:
        img_paths.extend(glob(os.path.join(path,"*" + ext)))

    img_paths = sorted(img_paths)

    return img_paths


def read_file_paths(folder):
    
    if folder[-1] != "/":
        folder += "/"
    
    return sorted(glob(folder + "*"))


def read_file_names(folder):
    return sorted(os.listdir(folder))


def json_load_classif_params(path):
    params = io_utils.json_load(path)

    # this bit is for keeping backwards compatibility with model params files that do not have the keep_aspect param
    if "keep_aspect" not in params:
        params.keep_aspect = False

    classIndex_2_class_conv = {int(k):v for k,v in params.classIndex_2_class.items()} #make keys int. json load problem
    params.classIndex_2_class = classIndex_2_class_conv
    params.img_size = tuple(params.img_size)
    
    return params


def roi_2_id(roi_proto):
    return "{}-{}-{}-{}".format(roi_proto.rect.tl.row,
                                roi_proto.rect.tl.col,
                                roi_proto.rect.br.row,
                                roi_proto.rect.br.col)


def compute_roi_area(xmax, xmin, ymax, ymin):
    """ Computes the area of a ROI represented by the coordinates given as arguments. """

    roi_width = xmax - xmin
    roi_height = ymax - ymin

    return roi_width * roi_height


def extract_cropped_roi(full_img, roi_proto):
    roi_id = roi_2_id(roi_proto)
    roi_img = full_img[roi_proto.rect.tl.row:roi_proto.rect.br.row,
              roi_proto.rect.tl.col:roi_proto.rect.br.col]

    return roi_img, roi_id

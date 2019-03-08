import logging
import os
import shutil

import numpy as np
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input,GlobalAveragePooling2D,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.optimizers import Adam
from tqdm import tqdm
import tensorflow as tf

import classification.scripts.utils as utils
import apollo_python_common.ftp_utils as ftp_utils
import apollo_python_common.io_utils as io_utils
from classification.scripts.constants import Column


def get_conv_model(searched_layer_name, size):
    model = InceptionV3(include_top=False, input_shape=(size[1], size[0]) + (3,), weights='imagenet')
    model_constrained = Model(model.input, model.get_layer(searched_layer_name).output)

    return model_constrained


def conv_block(x, nr_filters):
    x = Conv2D(nr_filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    return x


def ssdish_conv_block(input_conv, nr_filters, dropout_level):
    conv_layer = conv_block(input_conv, nr_filters)

    pred_layer = Dropout(dropout_level)(conv_layer)

    return conv_layer, pred_layer


def get_ssd_conv_model(input_shape, nr_conv_blocks, nr_filters, dropout_level, nr_classes):
    pred_list = []

    inp = Input(input_shape)
    x = inp
    x = BatchNormalization(axis=-1)(inp)

    prev_conv = x

    for index in range(nr_conv_blocks):
        prev_conv, current_pred = ssdish_conv_block(prev_conv, nr_filters, dropout_level)
        pred_list.append(current_pred)

    if len(pred_list) > 1:
        x = Concatenate()(pred_list)
    else:
        x = pred_list[0]
       
    x = Conv2D(nr_classes, (3, 3), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Activation('softmax')(x)

    model = Model(inp, predictions)

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def get_pred_df(model, data_df):
    test_conv_img_data = utils.numpify(data_df[Column.CONV_IMG_COL])

    predictions = model.predict(test_conv_img_data, verbose=1)

    data_df.loc[:, Column.PRED_COL] = pd.Series([pred for pred in predictions], index=data_df.index)
    data_df.loc[:, Column.CORRECT_COL] = data_df.apply(
        lambda row: 1 if np.argmax(row[Column.PRED_COL]) == np.argmax(row[Column.LABEL_COL]) else 0, axis=1)

    return data_df


def make_prediction_on_dataset(df_path_list, img_path_list, conv_path_list, model, nr_batches=None, with_img=False):
    logger = logging.getLogger(__name__)
    logger.info("Predicting...")

    batch_df_paths = []
    batch_img_paths = []
    batch_conv_paths = []

    for df_path, img_path, conv_path in zip(df_path_list, img_path_list, conv_path_list):
        split_batch_df_paths = sorted([os.path.join(df_path,batch_df_path) for batch_df_path in os.listdir(df_path)])
        split_batch_img_paths = sorted([os.path.join(img_path,batch_img_path) for batch_img_path in os.listdir(img_path)])
        split_batch_conv_paths = sorted([os.path.join(conv_path,batch_conv_path) for batch_conv_path in os.listdir(conv_path)])

        batch_df_paths += split_batch_df_paths
        batch_img_paths += split_batch_img_paths
        batch_conv_paths += split_batch_conv_paths

    if nr_batches is None:
        nr_batches = len(batch_df_paths)

    pred_df = []

    for batch_df_path, batch_img_path, batch_conv_path in tqdm(
            list(zip(batch_df_paths, batch_img_paths, batch_conv_paths))[:nr_batches]):

        if with_img:
            data_df_split = utils.merge_df_with_data(batch_df_path, batch_img_path, Column.IMG_COL)
            conv_data = np.load(batch_conv_path)
            data_df_split.loc[:, Column.CONV_IMG_COL] = pd.Series([c for c in conv_data], index=data_df_split.index)
        else:
            data_df_split = utils.merge_df_with_data(batch_df_path, batch_conv_path, Column.CONV_IMG_COL)

        pred_df_split = get_pred_df(model, data_df_split).drop(Column.CONV_IMG_COL, axis=1)

        pred_df.append(pred_df_split)

    pred_df = pd.concat(pred_df, axis=0)

    return pred_df


def load_model_from_json_and_weights(json_path, weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    return model


def save_model_to_json(model, json_path):
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)


def save_model_bundle(bundle_path, model, params):
    io_utils.create_folder(bundle_path)
    save_model_to_json(model, os.path.join(bundle_path, "model_structure.json"))
    model.save_weights(os.path.join(bundle_path, "model_weights.h5"))
    io_utils.json_dump(params, os.path.join(bundle_path, "model_params.json"))


def load_tf_model(bundle_path):
    weights_file = os.path.join(bundle_path,"model_trt_FP32.pb")

    with tf.Graph().as_default() as graph:
        with tf.gfile.GFile(weights_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="classif")
        predict_model = tf.Session(graph=graph)
    return predict_model


def load_keras_model(bundle_path, params):
    ssdish_model = load_model_from_json_and_weights(os.path.join(bundle_path,"model_structure.json"),
                                                    os.path.join(bundle_path,"model_weights.h5"))

    conv_model = get_conv_model(params.conv_layer_name, params.img_size)

    full_model = Sequential()
    full_model.add(conv_model)
    full_model.add(ssdish_model)

    return full_model


def load_model_bundle(bundle_path, tf_version):
    logger = logging.getLogger(__name__)
    logger.info("Building Model...")

    params = utils.json_load_classif_params(os.path.join(bundle_path,"model_params.json"))

    model = load_tf_model(bundle_path) if tf_version else load_keras_model(bundle_path, params)

    return model, params


def load_model_bundle_from_ftp(ftp_bundle_path, tf_version=False):
    local_bundle_path = "./ftp_bundle/"
    ftp_utils.copy_zip_and_extract(ftp_bundle_path, local_bundle_path)

    model, params = load_model_bundle(local_bundle_path, tf_version)

    shutil.rmtree(local_bundle_path)

    return model, params


def get_hydra_model(params, index_2_model, index_2_algorithm_name):
    hydra_body_model = get_conv_model(params.conv_layer_name, params.img_size)
    inp = Input((params.img_size[1], params.img_size[0]) + (3,))
    neck = hydra_body_model(inp)

    head_list = []
    for index, hydra_head_model in index_2_model.items():
        hydra_head_model_copy = Model(hydra_head_model.inputs, hydra_head_model.outputs,
                                      name=index_2_algorithm_name[index])
        head = hydra_head_model_copy(neck)
        head_list.append(head)

    hydra_full_model = Model(inp, head_list)
    hydra_full_model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return hydra_full_model

import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from tqdm import tqdm

import classification.scripts.utils as utils
from classification.scripts.constants import Column


def label2text(label, classIndex_2_class):
    return classIndex_2_class[np.argmax(label)]


def get_classification_report(pred_data_df, classIndex_2_class, nr_classes):
    ground_truth = [label2text(label, classIndex_2_class) for label in utils.numpify(pred_data_df[Column.LABEL_COL])]
    preds = [label2text(pred, classIndex_2_class) for pred in utils.numpify(pred_data_df[Column.PRED_COL])]

    return classification_report(ground_truth, preds,
                                 target_names=[classIndex_2_class[index] for index in range(nr_classes)], digits=3)


def get_confidence_accuracy(way_id_pred_df, conf_level):
    conf_preds_indexes = way_id_pred_df[Column.PRED_COL].apply(lambda pred: max(pred) >= conf_level)
    conf_df = way_id_pred_df.loc[conf_preds_indexes.tolist()]

    return compute_accuracy(conf_df)


def compute_accuracy(pred_df):
    nr_correct = -1

    try:
        nr_correct = pred_df.loc[:, Column.CORRECT_COL].value_counts()[1]
    except:
        nr_wrong = 0

    nr_wrong = -1

    try:
        nr_wrong = pred_df.loc[:, Column.CORRECT_COL].value_counts()[0]
    except:
        nr_wrong = 0

    accuracy = float(nr_correct) / max((nr_correct + nr_wrong), 1)

    return accuracy


def get_way_id_list(df_path):
    split_paths = sorted([split_path for split_path in os.listdir(df_path)])

    way_id_list = []

    for split_path in tqdm(split_paths):
        data_df_split = pd.read_pickle(df_path + split_path)

        way_id_list += data_df_split[Column.WAY_ID_COL].tolist()

    return way_id_list


def compute_same_wayid_not_same_wayid_accuracy(pred_data_df, train_df_path):
    train_way_ids = get_way_id_list(train_df_path)

    same_data_df = pred_data_df.loc[pred_data_df[Column.WAY_ID_COL].isin(train_way_ids)]
    not_same_data_df = pred_data_df.loc[~pred_data_df[Column.WAY_ID_COL].isin(train_way_ids)]

    print("Same Way Id Accuracy = {} made from {} examples".format(compute_accuracy(same_data_df), len(same_data_df)))
    print("Not Same Way Id Accuracy = {} made from {} examples".format(compute_accuracy(not_same_data_df),
                                                                       len(not_same_data_df)))


def get_img_id_list(df_path):
    split_paths = sorted([split_path for split_path in os.listdir(df_path)])

    way_id_list = []

    for split_path in tqdm(split_paths):
        data_df_split = pd.read_pickle(df_path + split_path)
        data_df_split = data_df_split.reset_index(drop=False)

        way_id_list += data_df_split[Column.IMG_NAME_COL].tolist()

    return way_id_list


def compute_same_imgid_not_same_imgid_accuracy(original_pred_data_df, train_df_path):
    img_id_list = get_img_id_list(train_df_path)

    pred_data_df = original_pred_data_df.copy().reset_index(drop=False)

    same_data_df = pred_data_df.loc[pred_data_df[Column.IMG_NAME_COL].isin(img_id_list)]
    not_same_data_df = pred_data_df.loc[~pred_data_df[Column.IMG_NAME_COL].isin(img_id_list)]

    print("Same Img Id Accuracy = {} made from {} examples".format(compute_accuracy(same_data_df), len(same_data_df)))
    print("Not Same Img Id Accuracy = {} made from {} examples".format(compute_accuracy(not_same_data_df),
                                                                       len(not_same_data_df)))


def get_accuracy_by_col(way_id_pred_df, group_by_col):
    bucket_2_accuracy = way_id_pred_df \
        .groupby(group_by_col) \
        .agg({Column.CORRECT_COL: 'mean'}) \
        .sort_values(Column.CORRECT_COL) \
        .to_dict()[Column.CORRECT_COL]

    bu_2_acc = zip(bucket_2_accuracy.keys(), bucket_2_accuracy.values())
    bu_2_acc = sorted(bu_2_acc, key=lambda tup: tup[0])

    keys = [key for key, _ in bu_2_acc]
    values = [value for _, value in bu_2_acc]

    return keys, values


def compute_accuracy_by_feature_threshold(pred_df, feature_name, feature_threshold, high_threshold=False):
    if high_threshold:
        valid_indexes = pred_df[feature_name].apply(lambda feature_value: feature_value <= feature_threshold)
    else:
        valid_indexes = pred_df[feature_name].apply(lambda feature_value: feature_value >= feature_threshold)

    valid_df = pred_df.loc[valid_indexes.tolist()]

    return compute_accuracy(valid_df)


def compute_recall_by_feature_threshold(pred_df, feature_name, feature_threshold, high_threshold=False):
    nr_total = len(pred_df)

    if high_threshold:
        valid_indexes = pred_df[feature_name].apply(lambda feature_value: feature_value <= feature_threshold)
    else:
        valid_indexes = pred_df[feature_name].apply(lambda feature_value: feature_value >= feature_threshold)

    nr_after_filter = float(Counter(valid_indexes)[True])

    return nr_after_filter / nr_total, nr_after_filter


def get_most_confident_images(data_df):
    data_df.loc[:, Column.PRED_CONF_COL] = data_df.loc[:, Column.PRED_COL].apply(lambda pred: max(pred))
    return data_df.sort_values(Column.PRED_CONF_COL, ascending=False)


def get_least_confident_images(data_df):
    data_df.loc[:, Column.PRED_CONF_COL] = data_df.loc[:, Column.PRED_COL].apply(lambda pred: max(pred))
    return data_df.sort_values(Column.PRED_CONF_COL)


def get_predictions_with_confidence_over(data_df, min_conf_level=None, max_conf_level=None):
    if min_conf_level is None:
        min_conf_level = 0.0

    if max_conf_level is None:
        max_conf_level = 1.0

    data_df.loc[:, Column.PRED_CONF_COL] = data_df.loc[:, Column.PRED_COL].apply(lambda pred: max(pred))
    return data_df[
        (data_df[Column.PRED_CONF_COL] >= min_conf_level) & (data_df[Column.PRED_CONF_COL] <= max_conf_level)]


def merge_way_with_img_predictions(pred_data_df, way_id_pred_df):
    joined_df = pd.merge(pred_data_df, way_id_pred_df,
                         left_on=[Column.WAY_ID_COL], right_on=[Column.WAY_ID_COL],
                         suffixes=("", "_way")
                         )

    joined_df = shuffle(joined_df, random_state=0)
    print("Pred shape = {}".format(pred_data_df.shape))
    print("Way shape = {}".format(way_id_pred_df.shape))
    print("Joined shape = {}".format(joined_df.shape))

    return joined_df


def get_predictions_within_nr_seqs_interval(data_df, min_nr_seqs=None, max_nr_seqs=None):
    if min_nr_seqs is None:
        min_nr_seqs = 0

    if max_nr_seqs is None:
        max_nr_seqs = 100000

    return data_df[(data_df[Column.NR_SEQS_COL] >= min_nr_seqs) & (data_df[Column.NR_SEQS_COL] <= max_nr_seqs)]


def check_data_integrity(train_data_df, test_data_df):
    print("-----Train Imgs-----")
    print(train_data_df[Column.LABEL_CLASS_COL].value_counts())
    print("-----Test Imgs-----")
    print(test_data_df[Column.LABEL_CLASS_COL].value_counts())

    print("-----Train Ways-----")
    print(train_data_df.groupby(Column.LABEL_CLASS_COL).agg({Column.WAY_ID_COL: "nunique"}))
    print("-----Test Ways-----")
    print(test_data_df.groupby(Column.LABEL_CLASS_COL).agg({Column.WAY_ID_COL: "nunique"}))

    train_way_ids = set(train_data_df[Column.WAY_ID_COL].tolist())
    test_way_ids = set(test_data_df[Column.WAY_ID_COL].tolist())

    print("Way ids intersection = %d" % len(train_way_ids.intersection(test_way_ids)))

    train_seq_ids = set(train_data_df[Column.SEQ_ID_COL].tolist())
    test_seq_ids = set(test_data_df[Column.SEQ_ID_COL].tolist())

    print("Seq ids intersection = %d" % len(train_seq_ids.intersection(test_seq_ids)))

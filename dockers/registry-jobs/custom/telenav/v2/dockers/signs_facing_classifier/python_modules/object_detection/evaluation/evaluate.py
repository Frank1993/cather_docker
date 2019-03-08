import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
import itertools
from collections import defaultdict
from collections import Counter
from collections import namedtuple
from multiprocessing import Pool
import multiprocessing

import apollo_python_common.proto_api as proto_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.rectangle import Rectangle
import orbb_definitions_pb2

SIZE_THRESHOLD = 25
IOU_THRESHOLD = 0.25

Roi = namedtuple('Roi', 'type tl_row tl_col br_row br_col')

MATCH_YES = "match_yes"
MATCH_NO = "match_no"
MATCH_MISCLASS = "match_misclass"

TP = "tp"
FP = "fp"
FN = "fn"
MC = "mc"

MATCH_STATUS_LIST = [TP, FP, FN, MC]

BIG = "big"
SMALL = "small"

GT_ROIS_COL = "gt_rois"
IMG_PROTO_COL = "img_proto"
IMG_NAME_COL = "img_name"
SELECTED_ROIS_COL = "selected_rois"
HIGH_CONF_ROIS_COL = "high_conf_rois"
NAIVE_MATCH_COL = "naive_match"
GT_ROI_COL = "gt_roi"
PRED_ROI_COL = "pred_roi"
PRED_ROI_ID_COL = "pred_roi_id"
GT_ROI_ID_COL = "gt_roi_id"
MATCH_COL = "match"
IS_MATCH_YES_COL = "is_match_yes"
IS_MATCH_NO_COL = "is_match_no"
IS_MATCH_MISCLASS_COL = "is_match_misclass"
PRED_ROIS_COL = "pred_rois"
IOU_COL = "iou"
ALREADY_SEEN_COL = "already_seen"


def get_class_name(index):
    return orbb_definitions_pb2._MARK.values_by_number[index].name


def get_all_roi_types(gt_df):
    roi_types = gt_df[GT_ROIS_COL].apply(lambda rois: [roi.type for roi in rois]).values
    roi_types = list(set([item for sublist in roi_types for item in sublist]))
    return roi_types


def get_roi_id(roi):
    return "{}_{}-{}-{}-{}".format(roi.type, roi.tl_row,
                                   roi.tl_col,
                                   roi.br_row,
                                   roi.br_col)


def is_big_roi(roi):
    rect = Rectangle(roi.tl_col, roi.tl_row, roi.br_col, roi.br_row)

    return rect.width() >= SIZE_THRESHOLD and rect.height() >= SIZE_THRESHOLD


def get_iou_for_pair(gt_roi, pred_roi):
    gt_rect = Rectangle(gt_roi.tl_col, gt_roi.tl_row, gt_roi.br_col, gt_roi.br_row)
    pred_rect = Rectangle(pred_roi.tl_col, pred_roi.tl_row, pred_roi.br_col, pred_roi.br_row)
    iou = gt_rect.intersection_over_union(pred_rect)
    return iou


def rois_match(row):
    roi_1 = row[GT_ROI_COL]
    roi_2 = row[PRED_ROI_COL]
    iou = row[IOU_COL]

    if iou < IOU_THRESHOLD:
        return MATCH_NO

    if roi_1.type != roi_2.type:
        return MATCH_MISCLASS

    return MATCH_YES


def get_metrics(metrics_dict):
    tp = metrics_dict[TP]
    fp = metrics_dict[FP]
    fn = metrics_dict[FN]
    mc = metrics_dict[MC]

    acc = round(float(tp) / (tp + fp + fn + mc), 4) if tp + fp + fn + mc != 0 else 0

    prec = round(float(tp) / (tp + fp), 4) if tp + fp != 0 else 0

    recall = round(float(tp) / (tp + fn), 4) if tp + fn != 0 else 0

    return (acc, prec, recall)


def pretty_print(metrics_dict, msg):
    acc, prec, recall = get_metrics(metrics_dict)

    print(msg)
    print("\tAccuracy  {}".format(acc))
    print("\tPrecision {}".format(prec))
    print("\tRecall    {}".format(recall))

    print("\tTP = {}, FP = {}, FN = {}, MC = {}\n".format(metrics_dict[TP],
                                                          metrics_dict[FP],
                                                          metrics_dict[FN],
                                                          metrics_dict[MC]))

    return acc


def write_metrics_to_csv(size_dict, output_path):
    all_roi_types = list(size_dict.keys())

    agg_metrics_dict = defaultdict(int)

    output_lines = ["roi_type,accuracy,precision,recall"]
    for roi_type in all_roi_types:
        class_metrics_dict = defaultdict(int)

        for match_status in MATCH_STATUS_LIST:
            match_status_count = sum_counts(size_dict, match_status, roi_type)
            class_metrics_dict[match_status] += match_status_count
            agg_metrics_dict[match_status] += match_status_count

        acc, prec, recall = get_metrics(class_metrics_dict)

        output_lines.append("{},{},{},{}".format(get_class_name(roi_type), acc, prec, recall))

    acc, prec, recall = get_metrics(agg_metrics_dict)
    output_lines.append("{},{},{},{}".format("TOTAL", acc, prec, recall))

    io_utils.create_folder(os.path.dirname(output_path))
    with open(output_path, 'w') as the_file:
        for line in output_lines:
            the_file.write('{}\n'.format(line))


def sum_counts(size_dict, target_match_status, roi_type):
    return sum([count for match_status, count in size_dict[roi_type].items() if match_status == target_match_status])


def print_metrics(size_dict, per_class_metrics=False):
    all_roi_types = list(size_dict.keys())

    agg_metrics_dict = defaultdict(int)

    for roi_type in all_roi_types:
        class_metrics_dict = defaultdict(int)

        for match_status in MATCH_STATUS_LIST:
            match_status_count = sum_counts(size_dict, match_status, roi_type)
            class_metrics_dict[match_status] += match_status_count
            agg_metrics_dict[match_status] += match_status_count

        if per_class_metrics:
            pretty_print(class_metrics_dict, get_class_name(roi_type))

    pretty_print(agg_metrics_dict, "Total ")


def convert_roi(roi):
    return Roi(roi.type, roi.rect.tl.row, roi.rect.tl.col, roi.rect.br.row, roi.rect.br.col)


def valid_roi_class(roi, selected_classes):
    if len(selected_classes) == 0:
        return True

    return get_class_name(roi.type) in selected_classes


def get_img_name(image_proto):
    return os.path.basename(image_proto.metadata.image_path)


def get_selected_rois(image_proto, selected_classes):
    return [roi for roi in image_proto.rois if valid_roi_class(roi, selected_classes)]


def is_confident_roi(roi, classes_thresholds):
    return roi.detections[0].confidence > classes_thresholds[get_class_name(roi.type)]


def get_high_conf_rois(rois, classes_thresholds):
    return [roi for roi in rois if is_confident_roi(roi, classes_thresholds)]


def convert_rois(rois):
    return [convert_roi(r) for r in rois]


def read_df(metadata_path, rois_col_name, selected_classes, classes_thresholds=defaultdict(int)):
    images = proto_api.read_imageset_file(metadata_path).images

    data_df = pd.DataFrame({IMG_PROTO_COL: images})

    data_df.loc[:, IMG_NAME_COL] = data_df.loc[:, IMG_PROTO_COL].apply(get_img_name)
    data_df.loc[:, SELECTED_ROIS_COL] = data_df.loc[:, IMG_PROTO_COL].apply(
        lambda rois: get_selected_rois(rois, selected_classes))
    data_df.loc[:, HIGH_CONF_ROIS_COL] = data_df.loc[:, SELECTED_ROIS_COL].apply(
        lambda rois: get_high_conf_rois(rois, classes_thresholds))
    data_df.loc[:, rois_col_name] = data_df.loc[:, HIGH_CONF_ROIS_COL].apply(convert_rois)

    data_df = data_df.drop([IMG_PROTO_COL, SELECTED_ROIS_COL, HIGH_CONF_ROIS_COL], axis=1)

    return data_df


def build_cross_product_df(gt_rois, pred_rois):
    cross_product = list(itertools.product(gt_rois, pred_rois))
    cross_product_df = pd.DataFrame(cross_product, columns=[GT_ROI_COL, PRED_ROI_COL])

    cross_product_df.loc[:, GT_ROI_ID_COL] = cross_product_df.loc[:, GT_ROI_COL].apply(get_roi_id)
    cross_product_df.loc[:, PRED_ROI_ID_COL] = cross_product_df.loc[:, PRED_ROI_COL].apply(get_roi_id)

    cross_product_df.loc[:, IOU_COL] = cross_product_df.apply(
        lambda r: get_iou_for_pair(r[GT_ROI_COL], r[PRED_ROI_COL]), axis=1)
    cross_product_df.loc[:, MATCH_COL] = cross_product_df.apply(lambda r: rois_match(r), axis=1)

    return cross_product_df


def merge_df(gt_df, pred_df):
    joined_df = pd.merge(gt_df, pred_df, how='left', on=IMG_NAME_COL)

    for row in joined_df.loc[joined_df.pred_rois.isnull(), PRED_ROIS_COL].index:
        joined_df.at[row, PRED_ROIS_COL] = []

    return joined_df


def roi_id_2_roi_type(roi_id):
    return int(roi_id.split("_")[0])


def build_count_dict(cross_product_df, count_dict):
    seen_pred_ids, seen_gt_ids = set(), set()

    draw_dict = defaultdict(lambda: defaultdict(list))

    for _, row in cross_product_df.sort_values(IOU_COL, ascending=False).iterrows():

        gt_roi_id = row[GT_ROI_ID_COL]
        pred_roi_id = row[PRED_ROI_ID_COL]
        pred_roi = row[PRED_ROI_COL]
        match_status = row[MATCH_COL]
        gt_roi = row[GT_ROI_COL]
        gt_roi_type = roi_id_2_roi_type(gt_roi_id)
        gt_roi_dim_status = get_roi_dim_status(gt_roi)

        if match_status == MATCH_NO:
            continue

        if gt_roi_id in seen_gt_ids or pred_roi_id in seen_pred_ids:
            continue

        if match_status == MATCH_YES:
            draw_dict[gt_roi_dim_status]["tp_pairs"].append((pred_roi, gt_roi))
            count_dict[gt_roi_dim_status][gt_roi_type][TP] += 1

        if match_status == MATCH_MISCLASS:
            draw_dict[gt_roi_dim_status]["mc_pairs"].append((pred_roi, gt_roi))
            count_dict[gt_roi_dim_status][gt_roi_type][MC] += 1

        seen_pred_ids.add(pred_roi_id)
        seen_gt_ids.add(gt_roi_id)

    not_seen_gt_df = cross_product_df.drop_duplicates(GT_ROI_ID_COL).copy()
    not_seen_gt_df.loc[:, ALREADY_SEEN_COL] = not_seen_gt_df.loc[:, GT_ROI_ID_COL] \
        .apply(lambda roi_id: roi_id in seen_gt_ids)
    not_seen_gt_df = not_seen_gt_df[~not_seen_gt_df[ALREADY_SEEN_COL]]

    for _, row in not_seen_gt_df.iterrows():
        gt_roi_type = roi_id_2_roi_type(row[GT_ROI_ID_COL])
        gt_roi_dim_status = get_roi_dim_status(row[GT_ROI_COL])
        count_dict[gt_roi_dim_status][gt_roi_type][FN] += 1
        draw_dict[gt_roi_dim_status]["fn_pairs"].append((row[GT_ROI_COL], row[GT_ROI_COL]))

    not_seen_pred_df = cross_product_df.drop_duplicates(PRED_ROI_ID_COL).copy()
    not_seen_pred_df.loc[:, ALREADY_SEEN_COL] = not_seen_pred_df.loc[:, PRED_ROI_ID_COL] \
        .apply(lambda roi_id: roi_id in seen_pred_ids)
    not_seen_pred_df = not_seen_pred_df[~not_seen_pred_df[ALREADY_SEEN_COL]]

    for _, row in not_seen_pred_df.iterrows():
        pred_roi_type = roi_id_2_roi_type(row[PRED_ROI_ID_COL])
        pred_roi_dim_status = get_roi_dim_status(row[PRED_ROI_COL])
        count_dict[pred_roi_dim_status][pred_roi_type][FP] += 1
        draw_dict[pred_roi_dim_status]["fp_pairs"].append((row[PRED_ROI_COL], row[PRED_ROI_COL]))

    return count_dict, draw_dict


def initiailize_count_dict(all_roi_types):
    count_dict = {}
    count_dict[BIG] = {}
    count_dict[SMALL] = {}

    for size, size_dict in count_dict.items():
        for roi_type in all_roi_types:
            size_dict[roi_type] = defaultdict(int)

    return count_dict


def compute_agg_count_dict(count_dict_list, all_roi_types):
    agg_count_dict = initiailize_count_dict(all_roi_types)

    for size_status in [BIG, SMALL]:
        for roi_type in all_roi_types:
            for pred_status in [TP, FP, FN, MC]:
                for count_dict in count_dict_list:
                    agg_count_dict[size_status][roi_type][pred_status] += count_dict[size_status][roi_type][pred_status]

    return agg_count_dict


def get_roi_dim_status(roi):
    return BIG if is_big_roi(roi) else SMALL


def compute_count_dict(data):
    row, all_roi_types = data

    count_dict = initiailize_count_dict(all_roi_types)

    gt_rois = row[GT_ROIS_COL]
    pred_rois = row[PRED_ROIS_COL]

    if len(gt_rois) == 0:
        for pred_roi in pred_rois:
            roi_dim_status = get_roi_dim_status(pred_roi)
            count_dict[roi_dim_status][pred_roi.type][FP] += 1
        return count_dict

    if len(pred_rois) == 0:
        for gt_roi in gt_rois:
            roi_dim_status = get_roi_dim_status(gt_roi)
            count_dict[roi_dim_status][gt_roi.type][FN] += 1
        return count_dict

    cross_product_df = build_cross_product_df(gt_rois, pred_rois)

    count_dict, _ = build_count_dict(cross_product_df, count_dict)

    return count_dict


def evaluate(gt_path, pred_path, selected_classes, classes_thresholds):
    gt_df = read_df(gt_path, GT_ROIS_COL, selected_classes)
    pred_df = read_df(pred_path, PRED_ROIS_COL, selected_classes, classes_thresholds)

    all_roi_types = get_all_roi_types(gt_df)

    joined_df = merge_df(gt_df, pred_df)

    rows = [r for _, r in list(joined_df.iterrows())]
    input_data = [(r, all_roi_types) for r in rows]

    nr_threads = multiprocessing.cpu_count() // 2
    count_dict_list = Pool(nr_threads).map(compute_count_dict, input_data)

    agg_count_dict = compute_agg_count_dict(count_dict_list, all_roi_types)

    return agg_count_dict


def get_selected_classes(selected_classes_path):
    return io_utils.json_load(selected_classes_path).selected_classes if selected_classes_path is not None else []


def get_class_thresh(class_thresholds_path):
    return io_utils.json_load(class_thresholds_path) if class_thresholds_path is not None else defaultdict(int)


def compute_evaluation_result(args):
    selected_classes = get_selected_classes(args.selected_classes_path)
    class_thresh = get_class_thresh(args.class_thresholds_path)

    agg_count_dict = evaluate(args.gt_path, args.pred_path, selected_classes, class_thresh)

    print_metrics(agg_count_dict["big"], per_class_metrics=True)

    if args.output_csv_path:
        write_metrics_to_csv(agg_count_dict["big"], args.output_csv_path)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-g", "--gt_path", help="path to ground truth roi", type=str,
                        required=True)
    parser.add_argument("-p", "--pred_path", help="path to pred roi", type=str,
                        required=True)
    parser.add_argument("-s", "--selected_classes_path", help="path file containing selected roi types", type=str,
                        required=False)
    parser.add_argument("-c", "--class_thresholds_path", help="path file containing selected class thresholds",
                        type=str,
                        required=False)
    parser.add_argument("-o", "--output_csv_path", help="output csv", type=str,
                        required=False)

    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    args = parse_arguments()

    try:
        compute_evaluation_result(args)
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

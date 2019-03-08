import sys
import os

sys.path.append(os.path.abspath('../../'))
sys.path.append(os.path.abspath('../../apollo_python_common/protobuf/'))
import apollo_python_common.proto_api as meta
import logging
import cv2
from tqdm import tqdm
import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
import apollo_python_common.image as image
import apollo_python_common.proto_api as proto_api
import itertools


def get_falses(gt_rois_lst, pred_rois_lst, min_size_px, iou_threshold=0.25, signpost_threshold_area=6000):
    gt_rois = {(proto_api.get_rect_from_roi(roi)): roi for roi in gt_rois_lst}
    pred_rois = {(proto_api.get_rect_from_roi(roi)): roi for roi in pred_rois_lst}
    # false_negatives
    found_gt_roi = set()
    for (gt_label, gt_rect), gt_roi in gt_rois.items():
        for (pred_label, pred_rect), pred_roi in pred_rois.items():
            if gt_label == pred_label and gt_rect.intersection_over_union(pred_rect) > iou_threshold:
                found_gt_roi.add((gt_label, gt_rect))
                break
    gt_not_found = set(gt_rois.keys()) - found_gt_roi
    false_negatives = list()
    for gt_label, gt_rect in gt_not_found:
        if gt_rect.width() > min_size_px and gt_rect.height() > min_size_px:
            false_negatives.append(gt_rois[(gt_label, gt_rect)])

    # false_positives
    found_pred_roi = set()
    for (pred_label, pred_rect), pred_roi in pred_rois.items():
        for (gt_label, gt_rect), gt_roi in gt_rois.items():
            if gt_label == pred_label and gt_rect.intersection_over_union(pred_rect) > iou_threshold:
                found_pred_roi.add((pred_label, pred_rect))
                break
    pred_not_found = set(pred_rois.keys()) - found_pred_roi
    false_positives = list()
    for pred_label, pred_rect in pred_not_found:
        if pred_rect.width() > min_size_px and pred_rect.height() > min_size_px:
            if pred_label == "SIGNPOST_GENERIC":
                if pred_rect.area() > signpost_threshold_area:
                    false_positives.append(pred_rois[(pred_label, pred_rect)])
            else:
                false_positives.append(pred_rois[(pred_label, pred_rect)])
    return false_positives, false_negatives


def get_mc(false_positives, false_negatives):
    misclassifieds = []
    iou_threshold = 0.25

    roi_cross_product = list(itertools.product(false_positives, false_negatives))
    for fp, fn in roi_cross_product:
        _, fp_rect = proto_api.get_rect_from_roi(fp)
        _, fn_rect = proto_api.get_rect_from_roi(fn)
        iou = fp_rect.intersection_over_union(fn_rect)
        if iou > iou_threshold:
            misclassifieds.append(fp)

    return misclassifieds


def draw_falses_on_img(image, false_positives, false_negatives, misclassifieds):
    for roi in false_positives:
        label_name, rect = proto_api.get_rect_from_roi(roi)
        color = (255, 0, 0)
        cv2.rectangle(image, (int(rect.xmin), int(rect.ymin)), (int(rect.xmax), int(rect.ymax)), color, 2)
        img_text = "FP " + label_name + ' %.3f' % roi.detections[0].confidence
        cv2.putText(image, img_text, (int(rect.xmin), int(rect.ymin) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    for roi in false_negatives:
        label_name, rect = proto_api.get_rect_from_roi(roi)
        color = (0, 255, 0)
        cv2.rectangle(image, (int(rect.xmin), int(rect.ymin)), (int(rect.xmax), int(rect.ymax)), color, 2)
        img_text = "FN " + label_name + ' %.3f' % roi.detections[0].confidence
        cv2.putText(image, img_text, (int(rect.xmin), int(rect.ymin) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    for roi in misclassifieds:
        label_name, rect = proto_api.get_rect_from_roi(roi)
        color = (0, 0, 255)
        cv2.rectangle(image, (int(rect.xmin), int(rect.ymin)), (int(rect.xmax), int(rect.ymax)), color, 2)
        img_text = "MC " + label_name + ' %.3f' % roi.detections[0].confidence
        cv2.putText(image, img_text, (int(rect.xmin), int(rect.ymin) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def get_filtered_below_threshold(rois, classes_thresholds):
    out_list = list()
    for roi in rois:
        label_name, rect = proto_api.get_rect_from_roi(roi)
        roi_confidence = roi.detections[0].confidence
        if roi_confidence >= classes_thresholds.get(label_name, 0.1):
            out_list.append(roi)
    return out_list


def main(conf):
    classes_thresholds = io_utils.json_load(conf.score_thresholds_file)
    gt_rois_dict = meta.create_images_dictionary(meta.read_imageset_file(conf.ground_truth_rois_file))
    pred_rois_dict = meta.create_images_dictionary(meta.read_imageset_file(conf.predictions_rois_file))
    all_files = set(gt_rois_dict.keys())

    pred_rois_dict = meta.get_confident_rois(pred_rois_dict, classes_thresholds)
    for file_name in tqdm(all_files):
        if file_name not in pred_rois_dict:
            logger.warning("{} was not found in predicted dataset".format(file_name))
            continue
        if file_name not in gt_rois_dict:
            logger.warning("{} was not found in ground truth dataset".format(file_name))
            continue

        false_positives, false_negatives = get_falses(gt_rois_dict[file_name],
                                                      pred_rois_dict[file_name],
                                                      conf.min_size,
                                                      conf.iou_threshold,
                                                      conf.signpost_threshold_area)

        misclassifieds = get_mc(false_positives, false_negatives)

        if "FP" not in conf.selected_errors:
            false_positives = list()

        if "FN" not in conf.selected_errors:
            false_negatives = list()

        if "MC" not in conf.selected_errors:
            misclassifieds = list()

        false_positives = __get_selected_rois(conf.selected_classes, false_positives)
        false_negatives = __get_selected_rois(conf.selected_classes, false_negatives)
        misclassifieds = __get_selected_rois(conf.selected_classes, misclassifieds)

        if len(false_positives) + len(false_negatives) + len(misclassifieds) > 0:
            img = image.get_bgr(os.path.join(conf.images_folder, file_name))
            for class_name in classes_thresholds.keys():
                false_positives_draw = __get_selected_rois([class_name], false_positives)
                false_negatives_draw = __get_selected_rois([class_name], false_negatives)
                misclassifieds_draw = __get_selected_rois([class_name], misclassifieds)
                if len(false_positives_draw) + len(false_negatives_draw) + len(misclassifieds_draw) > 0:
                    draw_falses_on_img(img, false_positives_draw, false_negatives_draw, misclassifieds_draw)
                    io_utils.create_folder(os.path.join(conf.output_folder, class_name))
                    cv2.imwrite(os.path.join(conf.output_folder, class_name, file_name), img)


def __get_selected_rois(selected_classes, all_rois):
    if len(selected_classes) == 0:
        # all classes
        return all_rois
    selected_rois = list()
    for roi in all_rois:
        label, rect = proto_api.get_rect_from_roi(roi)
        if label in selected_classes:
            selected_rois.append(roi)
    return selected_rois


if __name__ == "__main__":
    if __name__ == "__main__":
        log_util.config(__file__)
    logger = logging.getLogger(__name__)
    conf = io_utils.json_load('draw_bad_rois.json')
    io_utils.create_folder(conf.output_folder)
    main(conf)

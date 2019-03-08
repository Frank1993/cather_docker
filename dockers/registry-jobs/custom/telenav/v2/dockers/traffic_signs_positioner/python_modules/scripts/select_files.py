from tqdm import tqdm
import cv2
import os
import numpy as np
import logging
import apollo_python_common.proto_api as meta
import object_detection.roi_ssd.utils.roi_utils as roi_utils
import apollo_python_common.log_util as log_util

# Generates a sample train set from an existing train set (more balanced) removing exit signs marked as speed limit signs

MAX_ROIS_PER_CLASS = 600
IMAGES_FOLDER = '/data/datasets/st_tr_tl_sl_gw_23_11_2017/'
YELLOW_THRESHOLD = ([20, 90, 90], [60, 255, 255])
SHOW_IMAGE = False

def select_files():
    roi_stat = meta.check_imageset(os.path.join(IMAGES_FOLDER, 'rois.bin'))
    roi_stat_new = dict([(class_name, 0) for class_name, counts in roi_stat.items()])
    logger.info(roi_stat)

    roi_dict = meta.create_images_dictionary(meta.read_imageset_file(os.path.join(IMAGES_FOLDER, 'rois.bin')))
    logger.info('initial number of files {}'.format(len(roi_dict.keys())))
    count_yellow = 0

    roi_dict_selected = dict()
    for file_base_name, rois in tqdm(roi_dict.items()):
        included = False
        full_file_name = os.path.join(IMAGES_FOLDER, file_base_name)
        for roi in rois:
            class_name = meta.get_roi_type_name(roi.type)
            if roi_stat_new[class_name] < MAX_ROIS_PER_CLASS and os.path.isfile(full_file_name):
                included = True
                break
        if included:
            for roi in rois:
                _, rect_source = roi_utils.get_rect_from_roi(roi)
                class_name = meta.get_roi_type_name(roi.type)
                if 'SPEED_LIMIT_' in class_name or 'SL_' in class_name:
                    source_img = cv2.imread(full_file_name, cv2.IMREAD_COLOR)
                    crop_source_img = source_img[rect_source.ymin:rect_source.ymax, rect_source.xmin:rect_source.xmax]
                    hsv = cv2.cvtColor(crop_source_img, cv2.COLOR_BGR2HSV)
                    lower = np.array(YELLOW_THRESHOLD[0], dtype = "uint8")
                    upper = np.array(YELLOW_THRESHOLD[1], dtype = "uint8")
                    mask = np.clip(cv2.inRange(hsv, lower, upper), 0, 1)
                    if np.sum(mask)  > crop_source_img.shape[0] * crop_source_img.shape[1] * 0.50:
                        count_yellow += 1
                        included = False
                        if SHOW_IMAGE:
                            cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
                            cv2.imshow("image", crop_source_img)
                            cv2.waitKey()
                        break
        if included:
            roi_dict_selected[file_base_name] = rois
            for roi in rois:
                class_name = meta.get_roi_type_name(roi.type)
                roi_stat_new[class_name] = roi_stat_new[class_name] + 1

    logger.info('count_yellow {}'.format(count_yellow))
    meta.serialize_proto_instance(meta.create_imageset_from_dict(roi_dict_selected), './', 'new_rois')
    logger.info(meta.check_imageset('new_rois.bin'))
    logger.info('final number of files {}'.format(len(roi_dict_selected.keys())))


if __name__=="__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    try:
        select_files()
    except Exception as err:
        logger.error(err)






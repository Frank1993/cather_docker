import json
import imutils
import logging
import random
import multiprocessing
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict

import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from apollo_python_common.rectangle import Rectangle
import apollo_python_common.proto_api as proto_api
from scripts.utils import *


def get_panel_rectangle(sign_rect, panel_width, panel_height, panel_position):
    sign_middle = (sign_rect.xmin + sign_rect.xmax) // 2
    if panel_position == "UP":
        panel_rect = Rectangle(sign_middle - panel_width // 2, sign_rect.ymin - panel_height,
                               sign_middle + panel_width // 2, sign_rect.ymin)
    else:
        panel_rect = Rectangle(sign_middle - panel_width // 2, sign_rect.ymax,
                               sign_middle + panel_width // 2, sign_rect.ymax + panel_height)
    return panel_rect


def add_panel_for_sign(source_img_file, tmpl_img_file, sign_rect, panel_position, show_image):
    tmpl_img = cv2.imread(tmpl_img_file, cv2.IMREAD_UNCHANGED)  # Load with transparency
    source_img = cv2.imread(source_img_file, cv2.IMREAD_COLOR)

    roi_enlarge_fact = ROI_ENLARGE_FACTOR
    y_translation = int(Y_TRANSLATION_FACTOR * (sign_rect.ymax - sign_rect.ymin))
    rotation_angle = randint(-ROTATION_ANGLE_THR, +ROTATION_ANGLE_THR)

    sign_width = sign_rect.xmax - sign_rect.xmin
    source_width = int(roi_enlarge_fact * sign_width)
    resize_factor = source_width / tmpl_img.shape[1]
    source_height = int(tmpl_img.shape[0] * resize_factor)
    source_rect = get_panel_rectangle(sign_rect, source_width, source_height, panel_position)
    source_rect.ymin += y_translation
    source_rect.ymax += y_translation
    crop_source_img = source_img[source_rect.ymin:source_rect.ymax, source_rect.xmin:source_rect.xmax]

    tmpl_img = imutils.rotate_bound(tmpl_img, rotation_angle)
    tmpl_img = cv2.resize(tmpl_img, (crop_source_img.shape[1], crop_source_img.shape[0]), interpolation=cv2.INTER_AREA)
    b_channel1, g_channel1, r_channel1, alpha_channel1 = cv2.split(tmpl_img)

    tmpl_bgr = cv2.merge((b_channel1, g_channel1, r_channel1))
    tmpl_bgr = random_transform(tmpl_bgr)

    crop_sign_image = source_img[sign_rect.ymin:sign_rect.ymax, sign_rect.xmin:sign_rect.xmax]

    for i in range(5):
        new_img_avg = np.average(tmpl_bgr, axis=(0, 1))
        sign_img_avg = np.average(crop_sign_image, axis=(0, 1))
        avg_factor = new_img_avg / sign_img_avg
        np.clip(avg_factor, 0.95, 1.05, out=avg_factor)
        tmpl_bgr = np.clip(tmpl_bgr / avg_factor, 0, 255)

    b_channel, g_channel, r_channel = cv2.split(tmpl_bgr)
    np.clip(b_channel.astype(np.uint8), 0, 255, out=b_channel)
    np.clip(g_channel.astype(np.uint8), 0, 255, out=g_channel)
    np.clip(r_channel.astype(np.uint8), 0, 255, out=r_channel)
    b_channel = b_channel.astype(np.uint8)
    g_channel = g_channel.astype(np.uint8)
    r_channel = r_channel.astype(np.uint8)
    tmpl_img = cv2.merge((b_channel, g_channel, r_channel, alpha_channel1))
    new_img = blend_transparent(crop_source_img, tmpl_img)
    source_img[source_rect.ymin:source_rect.ymax, source_rect.xmin:source_rect.xmax] = new_img
    if show_image:
        cv2.namedWindow("image", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("image", source_img)
        cv2.waitKey()
    return source_img, source_rect


def make_template_similar(base_image_bgr, template_img_bgra):
    base_hsv = cv2.cvtColor(base_image_bgr, cv2.COLOR_BGR2HSV)  # convert it to hsv

    b_channel, g_channel, r_channel, a_channel = cv2.split(template_img_bgra)
    template_bgr = cv2.merge((b_channel, g_channel, r_channel))
    template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)  # convert it to hsv

    base_saturation_avg = np.mean(base_hsv[:, :, 1])
    template_saturation_img = template_hsv[:, :, 1]
    template_saturation_img = template_saturation_img[np.where(a_channel > 0)]
    template_saturation_avg = np.mean(template_saturation_img)
    saturation_difference = base_saturation_avg - template_saturation_avg
    template_hsv[:, :, 1] = np.clip(template_hsv[:, :, 1] + saturation_difference, 0, 255)

    base_brigthness_avg = np.mean(base_hsv[:, :, 2])
    template_brightness_img = template_hsv[:, :, 2]
    template_brightness_img = template_brightness_img[np.where(a_channel > 0)]
    template_brigthness_avg = np.mean(template_brightness_img)
    brigthness_difference = base_brigthness_avg - template_brigthness_avg
    template_hsv[:, :, 2] = np.clip(template_hsv[:, :, 2] + brigthness_difference, 0, 255)

    template_bgr = cv2.cvtColor(template_hsv, cv2.COLOR_HSV2BGR)
    new_b_channel, new_g_channel, new_r_channel = cv2.split(template_bgr)
    new_template_img_bgra = cv2.merge((new_b_channel, new_g_channel, new_r_channel, a_channel))
    return new_template_img_bgra


def substitute_tmpl_in_img(source_img, tmpl_img_file, source_rect, target_class_in_subst, show_image):
    roi_enlarge_fact = ROI_ENLARGE_FACTOR + 0.3 if target_class_in_subst == 'EXCLUSION_ACTIVATED_WRONG_WAY_US' else ROI_ENLARGE_FACTOR
    tmpl_img = cv2.imread(tmpl_img_file, cv2.IMREAD_UNCHANGED)  # Load with transparency

    x_enlarge = int(roi_enlarge_fact * (source_rect.xmax - source_rect.xmin))
    y_enlarge = int(roi_enlarge_fact * (source_rect.ymax - source_rect.ymin))
    y_translation = int(Y_TRANSLATION_FACTOR * (source_rect.ymax - source_rect.ymin))
    new_xmin = max(source_rect.xmin - x_enlarge, 0)
    new_ymin = max(source_rect.ymin - y_enlarge + y_translation, 0)
    new_xmax = min(source_rect.xmax + x_enlarge, source_img.shape[1] - 1)
    new_ymax = min(source_rect.ymax + y_enlarge + y_translation, source_img.shape[0] - 1)
    source_rect = Rectangle(new_xmin, new_ymin, new_xmax, new_ymax)
    crop_source_img = source_img[source_rect.ymin:source_rect.ymax, source_rect.xmin:source_rect.xmax]

    rotation_angle = randint(-ROTATION_ANGLE_THR, +ROTATION_ANGLE_THR)
    tmpl_img = imutils.rotate_bound(tmpl_img, rotation_angle)
    tmpl_img = cv2.resize(tmpl_img, (crop_source_img.shape[1], crop_source_img.shape[0]), interpolation=cv2.INTER_AREA)
    tmpl_img = make_template_similar(crop_source_img, tmpl_img)
    new_img = blend_transparent(crop_source_img, tmpl_img)
    source_img[source_rect.ymin:source_rect.ymax, source_rect.xmin:source_rect.xmax] = new_img

    if show_image:
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('image', 100, 100)
        cv2.imshow("image", cv2.resize(source_img, (1024, 1024)))
        cv2.waitKey()
    return source_img


def get_tmpl_files_per_class(folder):
    dict_tmpl_files_per_cl = dict()
    imgs_per_class_tmpl_folders = [d for d in os.listdir(folder) if
                                   os.path.isdir(os.path.join(folder, d))]

    for d in imgs_per_class_tmpl_folders:
        dict_tmpl_files_per_cl[d] = [os.path.join(folder, d, f) for f in
                                     os.listdir(os.path.join(folder, d)) if '.png' in f and not '._' in f]
    logger.info("\nTemplates:")
    logger.info(dict([(c, len(l)) for (c, l) in dict_tmpl_files_per_cl.items()]))
    logger.info(dict_tmpl_files_per_cl)
    logger.info("")
    return dict_tmpl_files_per_cl


def add_panel(data):
    logger = logging.getLogger(__name__)

    file_base_name, rois, dict_tmpl_files_per_class, subst_with_classes, panel_position, show_image = data

    roi_list = []
    try:
        full_file_name = os.path.join(IMAGES_FOLDER, file_base_name)
        new_file_name = os.path.join(IMAGES_OUT_FOLDER, file_base_name)
        if os.path.isfile(full_file_name):
            if len(rois) > 0:
                for roi in rois:
                    _, rect_source = proto_api.get_rect_from_roi(roi)
                    existent_class_in_subst = proto_api.get_roi_type_name(roi.type)
                    target_class_in_subst = subst_with_classes[random.randint(0, len(subst_with_classes) - 1)]
                    template_files = dict_tmpl_files_per_class[target_class_in_subst]
                    template_file_name = template_files[randint(0, len(template_files) - 1)]
                    print("Adding for sign {} an additional panel of class {} in file {} using template {}".format(
                        existent_class_in_subst,
                        target_class_in_subst,
                        file_base_name,
                        template_file_name))
                    new_img, new_roi = add_panel_for_sign(full_file_name,
                                                          template_file_name,
                                                          rect_source,
                                                          panel_position,
                                                          show_image)
                    roi.type = proto_api.get_roi_type_value(target_class_in_subst)
                    roi.rect.tl.col = new_roi.xmin
                    roi.rect.tl.row = new_roi.ymin
                    roi.rect.br.col = new_roi.xmax
                    roi.rect.br.row = new_roi.ymax
                    roi_list.append(roi)
                cv2.imwrite(new_file_name, new_img)
    except Exception as err:
        logger.error(err)
    return roi_list


def add_additional_panels(roi_dict, rois_file, dict_tmpl_files_per_class, subst_with_classes, panel_position, show_image):

    previous_dict = proto_api.create_images_dictionary(proto_api.read_imageset_file(rois_file))

    file_base_name_arr, rois_arr = zip(*(list(roi_dict.items())[:MAX_NR_OF_FILES]))
    dict_tmpl_files_per_class_arr = [dict_tmpl_files_per_class] * len(rois_arr)

    subst_with_classes_arr = [subst_with_classes] * len(rois_arr)
    panel_position_arr = [panel_position] * len(rois_arr)
    show_image_arr = [show_image] * len(rois_arr)

    threads_number = multiprocessing.cpu_count() // 2
    pool = Pool(threads_number)
    rois_list = pool.map(add_panel, zip(file_base_name_arr, rois_arr, dict_tmpl_files_per_class_arr,
                                        subst_with_classes_arr, panel_position_arr, show_image_arr))
    pool.close()

    new_roi_dict = defaultdict(list)
    for file_base_name, rois in zip(file_base_name_arr, rois_list):
        new_roi_dict[file_base_name] = rois

    merged_roi_dict = proto_api.merge_image_dictionaries(previous_dict, new_roi_dict)
    return merged_roi_dict


def substitute_rois(roi_dict, dict_tmpl_files_per_class, subst_in_classes, subst_with_classes, show_image):
    logger = logging.getLogger(__name__)
    subst_count = 0
    new_img = None
    not_found_files = []
    for file_base_name, rois in tqdm(roi_dict.items()):
        try:
            full_file_name = os.path.join(IMAGES_FOLDER, file_base_name)
            new_file_name = os.path.join(IMAGES_OUT_FOLDER, file_base_name)
            if os.path.isfile(full_file_name) and subst_count < MAX_NR_OF_FILES:
                new_img = cv2.imread(full_file_name, cv2.IMREAD_COLOR)
                for roi in rois:
                    sign_class = proto_api.get_roi_type_name(roi.type)
                    if sign_class not in subst_in_classes:
                        continue
                    _, rect_source = proto_api.get_rect_from_roi(roi)
                    target_class_in_subst = subst_with_classes[random.randint(0, len(subst_with_classes) - 1)]
                    template_files = dict_tmpl_files_per_class[target_class_in_subst]
                    template_file_name = template_files[randint(0, len(template_files) - 1)]
                    print("Substituting {} with {} in file {} using template {}".format(sign_class,
                                                                                        target_class_in_subst,
                                                                                        file_base_name,
                                                                                        template_file_name))
                    new_img = substitute_tmpl_in_img(new_img, template_file_name, rect_source,
                                                     target_class_in_subst, show_image)
                    roi.type = proto_api.get_roi_type_value(target_class_in_subst)
                subst_count += 1
                cv2.imwrite(new_file_name, new_img)
            else:
                not_found_files.append(file_base_name)
        except Exception as err:
            logger.error(err)
            not_found_files.append(file_base_name)

    for file_base_name in not_found_files:
        del roi_dict[file_base_name]
    print('subst_count {}'.format(subst_count))


def get_meta_dict(rois_file, selected_classes):
    logger = logging.getLogger(__name__)
    logger.info("Metadata:")
    logger.info(proto_api.check_imageset(rois_file))
    roi_dict = proto_api.get_filtered_imageset_dict(rois_file, selected_classes)
    return roi_dict


def change_class_names(rois_dict, replace_class_names):
    for file_base_name, rois in tqdm(rois_dict.items()):
        for roi in rois:
            sign_class = proto_api.get_roi_type_name(roi.type)
            if sign_class in list(replace_class_names.keys()):
                roi.type = proto_api.get_roi_type_value(replace_class_names[sign_class])
    return rois_dict


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # ********* PARAMETERS:
    with open('rois_substitute.json') as json_data_file:
        data = json.load(json_data_file)
    SUBST_IN_CLASSES = [str(val) for val in data['SUBST_IN_CLASSES']]
    SUBST_WITH_CLASSES = [str(val) for val in data['SUBST_WITH_CLASSES']]
    KEEP_CLASSES = [str(val) for val in data['KEEP_CLASSES']]
    REPLACE_CLASS_NAMES = data['REPLACE_CLASS_NAMES']
    IMGS_PER_CLASS_TMPL_FOLDER = data['IMGS_PER_CLASS_TMPL_FOLDER']
    IMAGES_FOLDER = data['IMAGES_FOLDER']
    ROIS_FILE = data['ROIS_FILE']
    IMAGES_OUT_FOLDER = data['IMAGES_OUT_FOLDER']
    SHOW_IMAGE = data['SHOW_IMAGE']
    ROI_ENLARGE_FACTOR = data['ROI_ENLARGE_FACTOR']
    Y_TRANSLATION_FACTOR = data['Y_TRANSLATION_FACTOR']
    ROTATION_ANGLE_THR = data['ROTATION_ANGLE_THR']
    MAX_NR_OF_FILES = data['MAX_NR_OF_FILES']
    IS_ADDITIONAL_PANEL = data['IS_ADDITIONAL_PANEL']
    PANEL_POSITION = data['PANEL_POSITION']

    io_utils.create_folder(IMAGES_OUT_FOLDER)
    NEEDED_CLASSES = SUBST_IN_CLASSES + KEEP_CLASSES
    roi_dict = get_meta_dict(ROIS_FILE, NEEDED_CLASSES)
    dict_tmpl_files_per_class = get_tmpl_files_per_class(IMGS_PER_CLASS_TMPL_FOLDER)
    if IS_ADDITIONAL_PANEL:
        roi_dict = add_additional_panels(roi_dict, ROIS_FILE, dict_tmpl_files_per_class,
                                         SUBST_WITH_CLASSES, PANEL_POSITION, SHOW_IMAGE)
    else:
        substitute_rois(roi_dict, dict_tmpl_files_per_class, SUBST_IN_CLASSES, SUBST_WITH_CLASSES, SHOW_IMAGE)
    if len(REPLACE_CLASS_NAMES) > 0:
        roi_dict = change_class_names(rois_dict=roi_dict, replace_class_names=REPLACE_CLASS_NAMES)
    metadata = proto_api.create_imageset_from_dict(roi_dict)
    proto_api.serialize_proto_instance(metadata, IMAGES_OUT_FOLDER, 'rois')
    logger.info(proto_api.check_imageset(os.path.join(IMAGES_OUT_FOLDER, 'rois.bin')))

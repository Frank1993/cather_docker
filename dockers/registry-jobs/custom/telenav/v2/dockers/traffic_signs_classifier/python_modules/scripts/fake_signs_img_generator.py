import os
import cv2
import apollo_python_common.proto_api as meta
import numpy as np
from random import randint, shuffle
import object_detection.roi_ssd.utils.roi_utils as roi_utils
from apollo_python_common.rectangle import Rectangle
import shutil
import imutils
import logging
from tqdm import tqdm
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
from scripts.utils import *

# ALL_CLASSES = SL_US_CLASSES
ALL_CLASSES = ((43, 'SL_STOP_SIGN'), (69, 'GIVE_WAY'))
SHOW_IMAGE = False
ROI_ENLARGE_FACTOR = 0.35
Y_TRANSLATION_FACTOR = 0.25
ROTATION_ANGLE_THR = 4


def substitute_tmpl_in_img(source_img_file, tmpl_img_file, source_rect, show_image):
    x_enlarge = int(ROI_ENLARGE_FACTOR * (source_rect.xmax - source_rect.xmin))
    y_enlarge = int(ROI_ENLARGE_FACTOR * (source_rect.ymax - source_rect.ymin))
    y_translation = int(Y_TRANSLATION_FACTOR * (source_rect.ymax - source_rect.ymin))
    source_rect = Rectangle(source_rect.xmin-x_enlarge, source_rect.ymin-y_enlarge + y_translation,
                            source_rect.xmax+x_enlarge, source_rect.ymax+y_enlarge + y_translation)
    source_img = cv2.imread(source_img_file, cv2.IMREAD_COLOR)
    crop_source_img = source_img[source_rect.ymin:source_rect.ymax, source_rect.xmin:source_rect.xmax]
    rotation_angle = randint(-ROTATION_ANGLE_THR, +ROTATION_ANGLE_THR)

    tmpl_img = cv2.imread(tmpl_img_file, -1)  # Load with transparency
    b_channel, g_channel, r_channel, alpha_channel = cv2.split(tmpl_img)
    tmpl_img_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    tmpl_img = imutils.rotate_bound(tmpl_img_RGBA, rotation_angle)
    tmpl_img = cv2.resize(tmpl_img, (crop_source_img.shape[1], crop_source_img.shape[0]), interpolation=cv2.INTER_CUBIC)
    b_channel1, g_channel1, r_channel1, alpha_channel1 = cv2.split(tmpl_img)

    tmpl_rgb = cv2.imread(tmpl_img_file, cv2.IMREAD_COLOR).astype(np.uint8)
    tmpl_rgb = imutils.rotate_bound(tmpl_rgb, rotation_angle)
    tmpl_rgb = random_transform(tmpl_rgb)
    tmpl_rgb = cv2.resize(tmpl_rgb, (crop_source_img.shape[1], crop_source_img.shape[0]))
    for i in range(5):
        new_img_avg = np.average(tmpl_rgb, axis=(0, 1))
        source_img_avg = np.average(crop_source_img, axis=(0, 1))
        avg_factor = new_img_avg / source_img_avg
        tmpl_rgb = np.clip(tmpl_rgb / avg_factor, 0, 255)

    # tmpl_img_avg = np.average(tmpl_rgb, axis=(0, 1))
    b_channel, g_channel, r_channel = cv2.split(tmpl_rgb)
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
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow('image', 100, 100)
        cv2.imshow("image", cv2.resize(source_img, (1024, 1024)))
        cv2.waitKey()
    return source_img


def get_tmpl_files_per_class(folder):
    dict_tmpl_files_per_cl = dict()
    imgs_per_class_tmpl_folders = [d for d in os.listdir(folder) if
                                   os.path.isdir(os.path.join(folder, d))]
    dict_all_classes = dict(ALL_CLASSES)
    for d in imgs_per_class_tmpl_folders:
        dict_tmpl_files_per_cl[d] = [os.path.join(folder, d, f) for f in
                              os.listdir(os.path.join(folder, d)) if '.png' in f and not '._' in f]
    dict_tmpl_files = dict([(dict_all_classes[int(d.replace('c_', ''))], l) for (d, l) in dict_tmpl_files_per_cl.items()])
    return dict_tmpl_files


def get_probabilities(roi_dict):
    dict_all_classes = dict(ALL_CLASSES)
    dict_roi_count_per_class = dict([(c, 0) for c in dict_all_classes.values()])
    count_rois = 0.0
    for file_base_name, rois in roi_dict.items():
        for roi in rois:
            dict_roi_count_per_class[dict_all_classes[roi.type]] += 1
            count_rois += 1

    avg_rois_per_class = float(count_rois // len(ALL_CLASSES))
    count_missing_rois = float(sum([(avg_rois_per_class - count_cl if count_cl<avg_rois_per_class  else 0) for (cl, count_cl) in dict_roi_count_per_class.items()]))

    print("dict_roi_count_per_class:")
    print(dict_roi_count_per_class)
    subst_in_probability = dict([(cl, (float(count_cl-avg_rois_per_class) / count_cl) if count_cl>=avg_rois_per_class else 0) for (cl, count_cl) in dict_roi_count_per_class.items()])
    print("subst_in_probability:")
    print(subst_in_probability)

    print("subst_with_probability:")
    subst_with_probability = dict([(cl,  ((avg_rois_per_class-count_cl) / count_missing_rois) if count_cl<avg_rois_per_class else 0) for (cl, count_cl) in dict_roi_count_per_class.items()])
    sum_prob = sum([p for (cl, p) in subst_with_probability.items()])
    print(subst_with_probability)
    return subst_in_probability, subst_with_probability


def balancing_classess(roi_dict, show_image):
    dict_classes = dict(ALL_CLASSES)
    dict_classes_back = dict([(v, k) for (k, v) in ALL_CLASSES])
    subst_count = 0
    new_img = None
    for file_base_name, rois in tqdm(roi_dict.items()):
        is_substituted = False
        full_file_name = os.path.join(IMAGES_FOLDER, file_base_name)
        new_file_name = os.path.join(IMAGES_OUT_FOLDER, file_base_name)
        if os.path.isfile(full_file_name):
            try:
                for roi in rois:
                    _, rect_source = roi_utils.get_rect_from_roi(roi)
                    existent_class_in_subst = dict_classes[roi.type]
                    if np.random.random_sample() < subst_in_probability[existent_class_in_subst]:
                        target_class_in_subst = np.random.choice(list(subst_with_probability.keys()),
                                                                 p=list(subst_with_probability.values()))
                        template_files = dict_tmpl_files_per_class[target_class_in_subst]
                        if len(template_files) > 0:
                            # have to do a substitution
                            tmpl_img_idx = randint(0, len(template_files) - 1)
                            template_file_name = template_files[tmpl_img_idx]
                            print(("Substituting {} with {} in file {} using template {}".format(existent_class_in_subst,
                                                                                                target_class_in_subst,
                                                                                                file_base_name,
                                                                                                template_file_name)))
                            new_img = substitute_tmpl_in_img(full_file_name, template_file_name, rect_source, show_image)
                            roi.type = dict_classes_back[target_class_in_subst]
                            subst_count += 1
                            is_substituted = True
                            break
            except Exception as ex:
                print(ex)
            if is_substituted:
                cv2.imwrite(new_file_name, new_img)
            else:
                print(("Copying file {}".format(full_file_name)))
                shutil.copy2(full_file_name, new_file_name)
    print(('subst_count {}'.format(subst_count)))


if __name__=="__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    # ********* PARAMETERS:
    dict_classes = dict(ALL_CLASSES)

    # **** B: Yield ****
    IMGS_PER_CLASS_TMPL_FOLDER = '/home/adrianm/data/imgs_per_class_tmpl_yield/'
    IMAGES_FOLDER = '/data/datasets/stop_signs_images/'
    ROIS_FILE = '/data/datasets/stop_signs_images/rois.bin'
    IMAGES_OUT_FOLDER = '/data/datasets/sl_transformed_yield/'

    logger.info("Metadata:")
    logger.info(meta.check_imageset(ROIS_FILE))
    roi_dict = meta.create_images_dictionary(meta.read_imageset_file(ROIS_FILE))
    for file_base_name, rois in roi_dict.items():
        remaining_rois = list()
        for roi in rois:
            if roi.type in dict_classes.keys():
                remaining_rois.append(roi)
        roi_dict[file_base_name] = remaining_rois

    io_utils.create_folder(IMAGES_OUT_FOLDER)
    dict_tmpl_files_per_class = get_tmpl_files_per_class(IMGS_PER_CLASS_TMPL_FOLDER)
    logger.info("\nTemplates:")
    logger.info(dict([(c, len(l)) for (c, l) in dict_tmpl_files_per_class.items()]))
    logger.info(dict_tmpl_files_per_class)
    logger.info("")
    subst_in_probability, subst_with_probability = get_probabilities(roi_dict)
    logger.info("")
    balancing_classess(roi_dict, SHOW_IMAGE)
    metadata = meta.create_imageset_from_dict(roi_dict)
    meta.serialize_proto_instance(metadata, IMAGES_OUT_FOLDER, 'rois')
    logger.info(meta.check_imageset(os.path.join(IMAGES_OUT_FOLDER, 'rois.bin')))

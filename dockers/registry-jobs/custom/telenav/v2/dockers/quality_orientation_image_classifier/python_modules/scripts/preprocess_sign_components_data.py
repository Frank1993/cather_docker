import os
import logging
import cv2
from tqdm import tqdm as tqdm
from collections import defaultdict
import orbb_definitions_pb2 as orbb_definitions
import apollo_python_common.log_util as log_util
import apollo_python_common.io_utils as io_utils
from apollo_python_common import proto_api as proto_api
from apollo_python_common import image as image_api
from apollo_python_common.rectangle import Rectangle


def __change_class_names(roi_dict, replacement_dict):
    for rois in tqdm(roi_dict.values()):
        for roi in rois:
            for component in roi.components:
                component_type = proto_api.get_component_type_name(component.type)
                if component_type in replacement_dict:
                    component.type = proto_api.get_component_type_value(replacement_dict[component_type])
    return roi_dict


def __get_roi_size(roi):
    roi_rect = Rectangle.initialize_with_proto_rect(roi.rect)
    return roi_rect.height(), roi_rect.width()


def __crop_image(rect, img_path):
    img = image_api.get_bgr(img_path)
    crop_img = img[rect.ymin: rect.ymax, rect.xmin: rect.xmax]
    return crop_img


def __create_crop(rect, image_name, crops_path, image_path, new_image_index):
    cropped_img = __crop_image(rect, os.path.join(image_path, image_name))
    name, extension = os.path.splitext(image_name)
    full_path = os.path.join(crops_path, f"{name}_{new_image_index}{extension}")
    cv2.imwrite(full_path, cropped_img)
    return full_path


def __check_text_component(roi):
    for component in roi.components:
        if component.type == orbb_definitions.GENERIC_TEXT:
            return True
    return False


def __get_imageset_dictionary(meta_file_path):
    imageset_proto = proto_api.read_imageset_file(meta_file_path)
    return proto_api.create_images_dictionary(imageset_proto)


def __add_components_to_proto(image_proto, component, roi_indx, signpost_rect):
    roi = image_proto.rois.add()
    roi.id = roi_indx
    roi.type = component.type
    roi.rect.tl.row = component.box.tl.row - signpost_rect.rect.tl.row
    roi.rect.tl.col = component.box.tl.col - signpost_rect.rect.tl.col
    roi.rect.br.row = component.box.br.row - signpost_rect.rect.tl.row
    roi.rect.br.col = component.box.br.col - signpost_rect.rect.tl.col
    roi.manual = 1
    return image_proto


def __get_classes_numbers(roi_dict):
    classes_numbers = defaultdict(int)
    for rois in tqdm(roi_dict.values()):
        for roi in rois:
            # to be remembered we only take rois which contain generic text inside of them
            if __check_text_component(roi):
                for component in roi.components:
                    classes_numbers[component.type] += 1
    return classes_numbers


def __create_new_proto(roi_dict, crops_path, images_path, min_size, min_nr_rois_inside_class):
    img_indx = 0
    classes_numbers = __get_classes_numbers(roi_dict)
    image_set = proto_api.get_new_imageset_proto()
    for file_name, rois in tqdm(roi_dict.items()):
        cropped_indx = 0
        for roi in rois:
            row, col = __get_roi_size(roi)
            # check if the signpost has components at all (not to be added if it doesen't)
            if len(roi.components) > 0 and __check_text_component(roi) and max(row, col) >= min_size:
                img_indx += 1
                cropped_indx += 1
                roi_indx = 0
                image_proto = image_set.images.add()
                roi_rect = Rectangle.initialize_with_proto_rect(roi.rect)
                cropped_image_name = __create_crop(roi_rect, file_name, crops_path, images_path, cropped_indx)
                image_proto = proto_api.add_mandatory_metadata_to_proto(
                    image_proto, cropped_image_name, file_name.split('_')[0], img_indx, 'US')
                image_proto = proto_api.add_mandatory_sensor_data_to_proto(
                    image_proto, 0.0, 0.0, 0.0, roi.rect.br.row - roi.rect.tl.row, roi.rect.br.col - roi.rect.tl.col, 0, 0, '')
                for component in roi.components:
                    component_rect = Rectangle.initialize_with_proto_rect(component.box)
                    if roi_rect.contains_rectangle(component_rect) and \
                            classes_numbers[component.type] >= min_nr_rois_inside_class:
                        roi_indx += 1
                        image_proto = __add_components_to_proto(image_proto, component, roi_indx, roi)

    proto_api.serialize_proto_instance(image_set, crops_path)


def main(conf):
    logger = logging.getLogger(__name__)
    logger.info("Extracting rois from {}".format(conf.input_meta_file_path))
    logger.info("Images path file: {}".format(conf.images_path))
    logger.info("Result proto file with new images path: {}".format(conf.preprocessed_rois_path))
    logger.info("Replacement classes values {}".format(conf.replace_class_names))
    logger.info("Min size value: {}".format(conf.min_size))
    logger.info("Minimum number of components to be kept from classes: {}".format(conf.min_nr_rois_inside_class))
    io_utils.create_folder(conf.preprocessed_rois_path)
    roi_dict = __get_imageset_dictionary(conf.input_meta_file_path)
    roi_dict = __change_class_names(roi_dict, conf.replace_class_names)
    __create_new_proto(roi_dict, conf.preprocessed_rois_path, conf.images_path, conf.min_size, conf.min_nr_rois_inside_class)


if __name__ == "__main__":
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    conf = io_utils.json_load('preprocess_sign_components_data.json')
    main(conf)
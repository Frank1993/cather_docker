import argparse

import apollo_python_common.proto_api as proto_api
import orbb_definitions_pb2
import apollo_python_common.io_utils as io_utils

from apollo_python_common.obj_detection_evaluator.model_statistics import ModelStatistics as ModelStatistics


def convert_rois_dictionary(rois_dictionary):
    dictionary = dict()
    roi_count = 0
    for file_name in rois_dictionary.keys():
        dictionary[file_name] = list()
        rois_list = rois_dictionary[file_name]
        for roi in rois_list:
            detection_id = roi.id if roi.id > 0 else roi_count
            new_detection = ModelStatistics.create_detection(roi_count, orbb_definitions_pb2.Mark.Name(roi.type),
                                                             roi.rect.tl.col, roi.rect.tl.row, roi.rect.br.col,
                                                             roi.rect.br.row)
            if new_detection is not None:
                dictionary[file_name].append(new_detection)
                roi_count += 1
    return dictionary


def get_data_dicts(expected_rois_file, actual_rois_file, selected_classes_file, classes_thresholds_file):
    expected_metadata = proto_api.read_imageset_file(expected_rois_file)
    actual_metadata = proto_api.read_imageset_file(actual_rois_file)

    expected_proto_dictionary = proto_api.create_images_dictionary(expected_metadata, True)
    actual_proto_dictionary = proto_api.create_images_dictionary(actual_metadata, False)

    if selected_classes_file:
        selected_classes = io_utils.json_load(selected_classes_file).selected_classes
        expected_proto_dictionary = proto_api.filter_rois_by_classes(selected_classes, expected_proto_dictionary)
        actual_proto_dictionary = proto_api.filter_rois_by_classes(selected_classes, actual_proto_dictionary)
    if classes_thresholds_file:
        classes_thresholds = io_utils.json_load(classes_thresholds_file)
        actual_proto_dictionary = proto_api.get_confident_rois(actual_proto_dictionary, classes_thresholds)

    expected_detection_dictionary = convert_rois_dictionary(expected_proto_dictionary)
    actual_detection_dictionary = convert_rois_dictionary(actual_proto_dictionary)

    return expected_detection_dictionary, actual_detection_dictionary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expected_rois_file",
                        type=str, required=True)
    parser.add_argument("-a", "--actual_rois_file",
                        type=str, required=True)
    parser.add_argument("-o", "--result_file",
                        type=str, required=False)
    parser.add_argument("-c", "--selected_classes_file",
                        type=str, required=False, default=None)
    parser.add_argument("-t", "--classes_thresholds_file",
                        type=str, required=False, default=None)
    parser.add_argument("-s", "--min_size", help="Min side size of the detection in pixels",
                        type=int, required=False, default=25)
    args = parser.parse_args()

    expected_detection_dictionary, actual_detection_dictionary = get_data_dicts(args.expected_rois_file,
                                                                                args.actual_rois_file,
                                                                                args.selected_classes_file,
                                                                                args.classes_thresholds_file)

    model_statistics = ModelStatistics(expected_detection_dictionary, actual_detection_dictionary, args.min_size)
    model_statistics.compute_model_statistics()
    model_statistics.output_statistics(args.result_file)


if __name__ == "__main__":
    main()


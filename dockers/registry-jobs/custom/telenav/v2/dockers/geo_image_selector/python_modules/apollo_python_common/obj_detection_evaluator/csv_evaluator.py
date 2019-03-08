import argparse
import pandas as pd

from apollo_python_common.obj_detection_evaluator.model_statistics import ModelStatistics as ModelStatistics


def create_detection_dictionary(csv_file):
    df = pd.read_csv(csv_file)
    dictionary = dict()
    for index, row in df.iterrows():
        file_name = row['img_name']
        if file_name not in dictionary:
            dictionary[file_name] = list()
        new_detection = ModelStatistics.create_detection(index, row['type'], row['tl_col'], row['tl_row'], row['br_col'], row['br_row'])
        if new_detection is not None:
            dictionary[file_name].append(new_detection)
    return dictionary


def main():
    MIN_SIZE=25
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--expected_csv_file",
                        type=str, required=True)
    parser.add_argument("-a", "--actual_csv_file",
                        type=str, required=True)
    parser.add_argument("-o", "--result_file",
                        type=str, required=True)
    args = parser.parse_args()

    expected_csv_file = args.expected_csv_file
    actual_csv_file = args.actual_csv_file
    result_file = args.result_file

    expected_detections_dict = create_detection_dictionary(expected_csv_file)
    actual_detections_dict = create_detection_dictionary(actual_csv_file)

    model_statistics = ModelStatistics(expected_detections_dict, actual_detections_dict, MIN_SIZE)
    model_statistics.compute_model_statistics()
    model_statistics.output_statistics(result_file)


if __name__ == "__main__":
    main()


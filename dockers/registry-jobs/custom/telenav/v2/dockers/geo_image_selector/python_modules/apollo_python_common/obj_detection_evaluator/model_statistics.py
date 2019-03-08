from collections import defaultdict
from collections import namedtuple
from operator import attrgetter

from apollo_python_common.rectangle import Rectangle as Rectangle


class Detection:
    def __init__(self, detection_id, detection_type, bbox):
        self.detection_id = detection_id
        self.detection_type = detection_type
        self.bbox = bbox

    def __str__(self):
        return "{} {} {}".format(self.detection_id, self.detection_type, self.bbox)


class Statistics(object):
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.miss_classified = 0
        self.iou_sum = 0
        self.iou = 0

    def recall(self):
        if self.true_positives + self.false_negatives > 0:
            return self.true_positives / (self.true_positives + self.false_negatives)
        else:
            return 0

    def precision(self):
        if (self.true_positives + self.false_positives) > 0:
            return self.true_positives / (self.true_positives + self.false_positives + self.miss_classified)
        else:
            return 0

    def accuracy(self):
        if self.true_positives + self.false_positives + self.miss_classified > 0:
            return self.true_positives / (
                    self.true_positives + self.false_positives + self.false_negatives + self.miss_classified)
        else:
            return 0

    def __str__(self):
        if self.true_positives:
            self.iou = self.iou_sum / self.true_positives
        return "tp = {} fp = {} fn = {} mc = {} precision = {:.4f} recall = {:.4f} accuracy = {:4f} iou = {:4f}" \
            .format(self.true_positives, self.false_positives, self.false_negatives, self.miss_classified,
                    self.precision(), self.recall(), self.accuracy(), self.iou)


class ModelStatistics:

    IOU_THRESHOLD = 0.25

    Match = namedtuple(
        'Match', ['expected_detection', 'actual_detection', 'iou'])

    def __init__(self, expected_detections, actual_detections, min_size):
        self.expected_detections = expected_detections
        self.actual_detections = actual_detections
        self.statistics = defaultdict(Statistics)
        self.min_size = min_size


    @staticmethod
    def rectangles_intersect(rect_a, rect_b):
        return rect_a.intersection_over_union(rect_b)

    @staticmethod
    def create_detection(detection_id, type_name, tl_col, tl_row, br_col, br_row):
        detection = None
        if tl_col >= 0 or tl_row >= 0 or br_col >= 0 or br_row >= 0:
            detection = Detection(detection_id, type_name, Rectangle(tl_col, tl_row, br_col, br_row))
        return detection

    def select_matched_pairs(self, expected_detections, actual_detections):
        matches = list()
        for expected_detection in expected_detections:
            for actual_detection in actual_detections:
                iou = self.rectangles_intersect(actual_detection.bbox, expected_detection.bbox)
                if iou > self.IOU_THRESHOLD:
                    matches.append(self.Match(expected_detection, actual_detection, iou))
        matches = sorted(matches, key=attrgetter('iou'), reverse=True)
        return matches

    def valid_rectangle_size(self, rect):
        return min(rect.width(), rect.height()) >= self.min_size

    def select_true_positives(self, match, processed_actual_detections, processed_expected_detections):
        if match.expected_detection.detection_type == match.actual_detection.detection_type and \
                self.valid_rectangle_size(match.expected_detection.bbox):
            self.statistics[match.expected_detection.detection_type].true_positives += 1
            self.statistics[match.expected_detection.detection_type].iou_sum += match.iou
            processed_actual_detections.add(match.actual_detection.detection_id)
            processed_expected_detections.add(match.expected_detection.detection_id)
        return processed_actual_detections, processed_expected_detections

    def select_miss_classified(self, match, processed_actual_detections, processed_expected_detections):
        if match.expected_detection.detection_type != match.actual_detection.detection_type and \
                self.valid_rectangle_size(match.expected_detection.bbox):
            self.statistics[match.expected_detection.detection_type].miss_classified += 1
            processed_actual_detections.add(match.actual_detection.detection_id)
            processed_expected_detections.add(match.expected_detection.detection_id)
        return processed_actual_detections, processed_expected_detections

    def select_false_negatives(self, expected_detections,
                               processed_expected_detections):
        for expected_detection in expected_detections:
            if expected_detection.detection_id not in processed_expected_detections and \
                    self.valid_rectangle_size(expected_detection.bbox):
                self.statistics[expected_detection.detection_type].false_negatives += 1

    def select_false_positives(self, actual_detections, processed_actual_detections):
        for actual_detection in actual_detections:
            if actual_detection.detection_id not in processed_actual_detections and\
                    self.valid_rectangle_size(actual_detection.bbox):
                self.statistics[actual_detection.detection_type].false_positives += 1

    def compute_model_statistics(self):
        missing_keys = set(self.actual_detections.keys()).difference(set(self.expected_detections.keys()))
        for k in missing_keys:
            self.expected_detections[k] = list()
        for expected_file in self.expected_detections.keys():
            if expected_file not in self.actual_detections.keys():
                print("Error miss match file " + expected_file)
                actual_image_detections = list()
            else:
                actual_image_detections = self.actual_detections[expected_file]
            expected_image_detections = self.expected_detections[expected_file]

            matches = self.select_matched_pairs(expected_image_detections, actual_image_detections)
            processed_actual_detections = set()
            processed_expected_detections = set()

            for match in matches:
                if match.actual_detection.detection_id not in processed_actual_detections and \
                        match.expected_detection.detection_id not in processed_expected_detections:
                    if not self.valid_rectangle_size(match.expected_detection.bbox):
                        processed_actual_detections.add(match.actual_detection.detection_id)
                        processed_expected_detections.add(match.expected_detection.detection_id)
                    processed_actual_detections, processed_expected_detections = self.select_true_positives \
                        (match, processed_actual_detections, processed_expected_detections)
                    processed_actual_detections, processed_expected_detections = self.select_miss_classified \
                        (match, processed_actual_detections, processed_expected_detections)
            self.select_false_negatives(expected_image_detections, processed_expected_detections)
            self.select_false_positives(actual_image_detections, processed_actual_detections)

        total_statistic = Statistics()
        for key, statistic in self.statistics.items():
            if key not in ['INVALID']:
                total_statistic.true_positives += statistic.true_positives
                total_statistic.false_positives += statistic.false_positives
                total_statistic.false_negatives += statistic.false_negatives
                total_statistic.miss_classified += statistic.miss_classified
                total_statistic.iou_sum += statistic.iou_sum
        self.statistics["Total"] = total_statistic

    def output_statistics(self, result_file):
        with open(result_file, "w") as file:
            for key, statistics_element in self.statistics.items():
                statistic_line = "{} {}".format(key, statistics_element)
                print(statistic_line)
                file.write(statistic_line + '\n')




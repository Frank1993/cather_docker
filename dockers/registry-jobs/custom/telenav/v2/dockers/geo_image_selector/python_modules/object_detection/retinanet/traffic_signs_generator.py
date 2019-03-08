import numpy as np
import logging
import os.path
from collections import defaultdict
from PIL import Image
from object_detection.keras_retinanet.preprocessing.generator import Generator
import apollo_python_common.image
import apollo_python_common.proto_api as meta
from tqdm import tqdm
from scripts.traffic_signs_types import TrafficSignsTypes
from scripts.sign_components_types import SignComponentsTypes


class TrafficSignsGenerator(Generator):
    def __init__(
            self,
            base_dir,
            transform_generator,
            is_for_signpost_components,
            **kwargs
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing TrafficSigns dataset from'.format(base_dir))
        self.image_names = []
        self.image_data = {}
        self.base_dir = base_dir
        if is_for_signpost_components:
            self.rois_labels = SignComponentsTypes(os.path.join(self.base_dir, 'rois.bin'))
        else:
            self.rois_labels = TrafficSignsTypes(os.path.join(self.base_dir, 'rois.bin'))
        self.image_names = self.__get_image_names()
        self.image_data = self.get_image_data()
        self.logger.info("Classes: {}".format(self.captions()))
        super().__init__(transform_generator, **kwargs)
        self.logger.info('Dataset was initialised.')

    def __get_image_names(self):
        out_list = list()
        for filename in tqdm(self.rois_labels.get_image_names()):
            full_path = os.path.join(self.base_dir, filename)
            if os.path.isfile(full_path) and TrafficSignsGenerator.is_valid_image(full_path):
                out_list.append(filename)
        return out_list

    @staticmethod
    def is_valid_image(file_name):
        try:
            Image.open(file_name)
            return True
        except:
            logging.info('Image {0} is invalid. It was eliminated.'.format(file_name))
            return False

    def rois_dict(self):
        return self.rois_labels.rois_dict

    def classes(self):
        return self.rois_labels.dict_idx_type

    def labels(self):
        return self.rois_labels.labels()

    def captions(self):
        return self.rois_labels.captions()

    def get_image_data(self):
        result = defaultdict(list)
        for img_file in self.image_names:
            rois = self.rois_dict()[img_file]
            for roi in rois:
                result[img_file].append({'x1': roi.rect.tl.col, 'x2': roi.rect.br.col,
                                         'y1': roi.rect.tl.row, 'y2': roi.rect.br.row,
                                         'class': roi.type})
        return result

    def size(self):
        return len(self.image_names)

    def num_classes(self):
        return self.rois_labels.num_classes()

    def name_to_label(self, name):
        return self.rois_labels.name_to_label(name)

    def label_to_name(self, label):
        return self.rois_labels.label_to_name(label)

    def image_path(self, image_index):
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        return apollo_python_common.image.get_aspect_ratio(self.image_path(image_index))

    def image_size(self, image_index):
        return apollo_python_common.image.get_size(self.image_path(image_index))

    def load_image(self, image_index):
        img = apollo_python_common.image.get_bgr(self.image_path(image_index))
        return img

    def load_annotations(self, image_index):
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes


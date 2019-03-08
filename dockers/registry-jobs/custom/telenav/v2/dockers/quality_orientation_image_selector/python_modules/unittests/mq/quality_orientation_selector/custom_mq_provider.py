import random

import apollo_python_common.proto_api as proto_api
from sklearn.utils import shuffle
from unittests.mq.components.abstract_mq_provider import AbstractMQProvider


class CustomMQProvider(AbstractMQProvider):

    def __init__(self, config, nr_good_protos, nr_bad_protos):
        self.nr_good_protos = nr_good_protos
        self.nr_bad_protos = nr_bad_protos
        super().__init__(config)

    def sum_to_one(self, n):
        values = [0.0, 1.0] + [random.random() for _ in range(n - 1)]
        values.sort()
        return [values[i + 1] - values[i] for i in range(n)]

    def get_good_quality_pred(self):
        confs = sorted(self.sum_to_one(2))
        return {"good": confs[-1],
                "bad": confs[0]
                }

    def get_bad_quality_pred(self):
        confs = sorted(self.sum_to_one(2))
        return {"good": confs[0],
                "bad": confs[-1]
                }

    def get_good_orientation_pred(self):
        confs = sorted(self.sum_to_one(4))
        best_conf = confs[-1]
        rest_confs = shuffle(confs[:-1])

        return {"up": best_conf,
                "down": rest_confs[0],
                "left": rest_confs[1],
                "right": rest_confs[2]
                }

    def get_bad_orientation_pred(self):
        confs = sorted(self.sum_to_one(4))
        worst_conf = confs[0]
        rest_confs = shuffle(confs[1:])

        return {"up": worst_conf,
                "down": rest_confs[0],
                "left": rest_confs[1],
                "right": rest_confs[2]
                }

    def get_good_preds(self):
        quality_dict_preds = self.get_good_quality_pred()
        orientation_dict_preds = self.get_good_orientation_pred()

        return {"image_quality": quality_dict_preds,
                "image_orientation": orientation_dict_preds,
                }

    def get_bad_preds(self):

        bad_reason = random.randint(1, 3)

        if bad_reason == 1:
            quality_dict_preds = self.get_bad_quality_pred()
            orientation_dict_preds = self.get_good_orientation_pred()

        if bad_reason == 2:
            quality_dict_preds = self.get_good_quality_pred()
            orientation_dict_preds = self.get_bad_orientation_pred()

        if bad_reason == 3:
            quality_dict_preds = self.get_bad_quality_pred()
            orientation_dict_preds = self.get_bad_orientation_pred()

        return {"image_quality": quality_dict_preds,
                "image_orientation": orientation_dict_preds,
                }

    def get_image_proto_list_with(self, pred_dict_list):

        image_proto_list = []

        for pred_dict in pred_dict_list:
            image_proto = proto_api.get_new_image_proto("-1", -1, "/test/path/", "US", -1, -1, False)

            for algorithm, class_pred_dict in pred_dict.items():

                classif_prediction_proto = image_proto.features.classif_predictions.add()
                classif_prediction_proto.algorithm_version = "-1"
                classif_prediction_proto.algorithm = algorithm

                for class_name, conf in class_pred_dict.items():
                    classif_prediction_proto_class = classif_prediction_proto.pred_classes.add()
                    classif_prediction_proto_class.class_name = class_name
                    classif_prediction_proto_class.confidence = round(float(conf), 4)

                classif_prediction_proto.chosen_pred_class_name = sorted(class_pred_dict.items(),key = lambda t: -t[1])[0][0]
                
            image_proto = image_proto.SerializeToString()
            image_proto_list.append(image_proto)

        return image_proto_list

    def get_proto_list(self):

        good_list = [self.get_good_preds() for _ in range(self.nr_good_protos)]
        bad_list = [self.get_bad_preds() for _ in range(self.nr_bad_protos)]

        pred_dict_list = shuffle(good_list + bad_list)

        image_proto_list = self.get_image_proto_list_with(pred_dict_list)

        return image_proto_list


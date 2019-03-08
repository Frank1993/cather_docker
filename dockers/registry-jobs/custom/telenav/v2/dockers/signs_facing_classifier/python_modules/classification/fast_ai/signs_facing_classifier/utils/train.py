import argparse
import logging

from fastai import *
from fastai.vision import *

import classification.fast_ai.signs_facing_classifier.utils.constants as const
from apollo_python_common import log_util, io_utils
from classification.fast_ai.signs_facing_classifier.utils import model_common


class NetworkTrainer:
    """ Class for training a classification network from a source folder, based  """

    def __init__(self, params):
        self.params = params
        self.learner = self._create_learner()

    def _create_learner(self):
        """ Create a FastAI learner with required parameters for training over a directory o images. """
        transforms = get_transforms(max_rotate=self.params.tfms_max_rotate, max_warp=self.params.tfms_max_warp,
                                    flip_vert=self.params.tfms_flip_vert, do_flip=self.params.tfms_do_flip)
        data = ImageDataBunch.from_folder(self.params.imgs_dir, ds_tfms=transforms, padding_mode="zeros",
                                          bs=self.params.batch_size, size=self.params.image_size).normalize(
            imagenet_stats)

        logger.info("learner classes: {} ".format(data.classes))
        io_utils.json_dump({const.LABEL_LIST: data.classes}, self.params.label_list_file)

        return create_cnn(data, const.MODEL_DICT[self.params.backbone_model], metrics=accuracy)

    def train_network(self):
        """ Trains a classification network using FastAI with a given backbone and the internal learner. """
        logger.info("started training classifier...")

        self.learner.fit_one_cycle(self.params.epochs, self.params.frozen_lr)
        self.learner.save(self.params.frozen_backbone_model)

        self.learner.unfreeze()
        self.learner.fit_one_cycle(self.params.unfreeze_epochs,
                                   slice(self.params.unfreeze_lr1, self.params.unfreeze_lr2),
                                   self.params.pct_start)
        self.learner.save(self.params.final_model)

        logger.info("finished training classifier.")
        precision, confusion_matrix = model_common.model_stats(self.learner)
        logger.info("precision: {} /n confusion_matrix: {} /n".format(precision, confusion_matrix))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--train_cfg_json", help="path to json containing training params",
                        type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    log_util.config(__file__)
    logger = logging.getLogger(__name__)

    args = parse_arguments()
    train_config = io_utils.json_load(args.train_cfg_json)

    try:
        trainer = NetworkTrainer(train_config)
        trainer.train_network()
    except Exception as err:
        logger.error(err, exc_info=True)
        raise err

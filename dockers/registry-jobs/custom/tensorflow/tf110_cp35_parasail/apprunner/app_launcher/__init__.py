import logging

from app_launcher.pai_tensorflow_launcher import PAITensorFlowLauncher
from app_launcher.philly_tensorflow_launcher import PhillyTensorFlowLauncher
from app_launcher.tensorflow_launcher import TensorFlowLauncher
from app_launcher.trainstation_tensorflow_launcher import TrainStationTensorFlowLauncher
from shared import consts, util


def run(args, argv):
    launcher = create_launcher(args, argv)
    if launcher is None:
        logging.error("Failed to create launcher")
        return None

    logging.info("Launcher=%s", launcher.__class__.__name__)
    return launcher.run()


def create_launcher(args, argv):
    launcher = None

    app_type = util.get_arg_value(args, consts.ARG_APP_TYPE)
    if app_type == "tensorflow":
        launcher = create_tf_launcher(args, argv)

    return launcher


def create_tf_launcher(args, argv):
    platform = util.get_arg_value(args, consts.ARG_HOST_PLATFORM)
    if platform == consts.PHILLY:
        launcher = PhillyTensorFlowLauncher(args, argv)
    elif platform == consts.TRAINSTATION:
        launcher = TrainStationTensorFlowLauncher(args, argv)
    elif platform == consts.PAI:
        launcher = PAITensorFlowLauncher(args, argv)
    else:
        raise NotImplementedError("No launcher implemented for platform {}".format(platform))
    return launcher

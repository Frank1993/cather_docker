import logging
import os

from runtime_builder.python_runtime_builder import PythonRuntimeBuilder
from shared import util, consts


def run(args):
    builder = create_builder(args)
    if builder is None:
        return None

    logging.info("Builder=%s", builder.__class__.__name__)
    return builder.run()


def create_builder(args):
    src_dir = os.path.abspath(util.get_arg_value(args, consts.ARG_BUILD_SOURCE))
    path = os.path.join(src_dir, consts.RUNTIME_INI)

    if not os.path.isfile(path):
        logging.error("%s is missing: %s", consts.RUNTIME_INI, path)
        return None

    builder = None
    config = util.parse_ini(path)

    type = config["Runtime"]["Type"]
    if type == "Python":
        builder = PythonRuntimeBuilder(args)

    return builder

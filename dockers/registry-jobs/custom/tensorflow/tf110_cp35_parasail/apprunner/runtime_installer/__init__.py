import logging
import os

from runtime_installer.python_runtime_installer import PythonRuntimeInstaller
from shared import util, consts


def run(args):
    installer = create_installer(args)
    if installer is None:
        return None

    logging.info("Installer=%s", installer.__class__.__name__)
    return installer.run()


def create_installer(args):
    src_dir = os.path.abspath(util.get_arg_value(args, consts.ARG_INSTALL_SOURCE))
    path = os.path.join(src_dir, consts.RUNTIME_INI)

    if not os.path.isfile(path):
        logging.error("%s is missing", consts.RUNTIME_INI)
        return None

    installer = None
    config = util.parse_ini(path)

    type = config["Runtime"]["Type"]
    if type == "Python":
        installer = PythonRuntimeInstaller(args)

    return installer

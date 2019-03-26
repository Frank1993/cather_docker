import logging
import os

import repository
from shared import consts, util, fsutil


class RuntimeLoader:
    def __init__(self, args):
        self.args = args

    def run(self):
        if not self.init():
            return False

        if not self.load():
            return False

        logging.info("Runtime is loaded")
        return True

    def init(self):
        self.host_platform = self.get_arg_value(consts.ARG_HOST_PLATFORM)
        self.app_type = self.get_arg_value(consts.ARG_APP_TYPE)

        self.working_root = util.get_working_root(self.args)
        self.data_root = util.get_data_root(self.args)
        self.runtime_staging_dir = os.path.join(self.working_root, consts.RUNTIME_STAGING_DIR_NAME)

        self.os_name = util.get_os_name()
        self.architecture = util.get_cpu_architecture()

        self.runtime_source = self.get_runtime_source()
        if not self.runtime_source:
            return False

        return True

    def get_runtime_source(self):
        runtime_source = self.get_arg_value(consts.ARG_RUNTIME_SOURCE)
        if runtime_source is not None:
            return fsutil.abspath(runtime_source)

        runtime_binary = self.get_arg_value(consts.ARG_RUNTIME_BINARY)
        runtime_name = self.get_arg_value(consts.ARG_RUNTIME_NAME)
        runtime_version = self.get_arg_value(consts.ARG_RUNTIME_VERSION)

        if not runtime_name:
            (runtime_name, runtime_version) = self.parse_runtime_config(runtime_binary, runtime_version)
            if not runtime_name:
                return False

        if not runtime_version:
            runtime_version = consts.LATEST

        logging.info("Runtime is [%s, %s]", runtime_name, runtime_version)

        repo = repository.get_repository("Runtime", self.data_root)
        runtime_source = repo.get_item_directory(runtime_name, runtime_version, self.os_name, self.architecture)
        logging.info("Runtime source is %s", runtime_source)

        return runtime_source

    def parse_runtime_config(self, runtime_binary, runtime_version):
        if not runtime_binary:
            runtime_binary = "default"

        logging.info("Parse config for runtime binary %s", runtime_binary)
        dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir, self.app_type + ".ini")

        if not os.path.isfile(path):
            logging.info("Runtime config is missing, use runtime_binary as runtime_name: %s", runtime_binary)
            return runtime_binary, runtime_version

        config = util.parse_ini(path)
        if not self.host_platform in config:
            logging.info("No section found for host platform %s, use runtime_binary as runtime_name: %s",
                         self.host_platform, runtime_binary)
            return runtime_binary, runtime_version

        section = config[self.host_platform]
        setting = section.get(runtime_binary, None)

        if not setting:
            logging.info("Failed to find runtime_name, use runtime_binary as runtime_name: %s", runtime_binary)
            return runtime_binary, runtime_version

        strs = setting.split(',', 1)
        runtime_name = strs[0].strip()
        if len(strs) > 1:
            runtime_version = strs[1].strip()

        return runtime_name, runtime_version

    def load(self):
        logging.info("Copy runtime from %s to %s", self.runtime_source, self.runtime_staging_dir)

        if not fsutil.directory_exists(self.runtime_source):
            logging.error("Runtime source directory %s does not exist", self.runtime_source)
            return False

        util.recreate_directory(self.runtime_staging_dir)
        fsutil.copy_directory(self.runtime_source, self.runtime_staging_dir)

        zip_path = os.path.join(self.runtime_staging_dir, consts.RUNTIME_ZIP_NAME)
        if os.path.isfile(zip_path):
            logging.info("Unzip runtime")
            util.unzip(zip_path, self.runtime_staging_dir)
            util.delete_file(zip_path)

        return True

    def get_arg_value(self, arg, default=None):
        return util.get_arg_value(self.args, arg, default)

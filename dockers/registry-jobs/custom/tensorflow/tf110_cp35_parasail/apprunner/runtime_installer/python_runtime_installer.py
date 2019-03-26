import logging
import os

from shared import util, consts


class PythonRuntimeInstaller:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.init()
        self.install()
        return True

    def init(self):
        self.host_platform = self.get_arg_value(consts.ARG_HOST_PLATFORM)
        self.os_name = util.get_os_name()
        self.source_dir = os.path.abspath(self.get_arg_value(consts.ARG_INSTALL_SOURCE))
        self.target_dir = os.path.abspath(self.get_arg_value(consts.ARG_INSTALL_TARGET))
        self.python_dir = os.path.join(self.target_dir, "python")
        self.env_dir = os.path.join(self.target_dir, "env")

        config_path = os.path.join(self.source_dir, consts.RUNTIME_INI)
        config = util.parse_ini(config_path)

        section = config["Runtime"]
        self.architecture = section.get("Architecture")

        section = config["Python"]
        self.python_version = section.get("Version")
        self.python_dist = section.get("Distribution")
        self.requirements_path = os.path.join(self.source_dir, consts.REQUIREMENTS_TXT)
        self.copy_site_packages = (section.get("CopySitePackages", "False") == "True")

        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            self.conda_spec_path = os.path.join(self.source_dir, consts.CONDA_SPEC_TXT)
            self.required_conda_packages = [name.strip() for name in section.get("RequiredCondaPackages", "").split(',')
                                            if name]
            if self.copy_site_packages:
                raise RuntimeError("Copy site packages does not support anaconda")

    def install(self):
        logging.info("Copy runtime to target")
        util.recreate_directory(self.target_dir)
        dirs = next(os.walk(self.source_dir))[1]
        for dir_name in dirs:
            if dir_name not in ["packages", "python", "env"]:
                util.copy_directory(os.path.join(self.source_dir, dir_name), os.path.join(self.target_dir, dir_name))
        util.copy_directory(self.source_dir, self.target_dir, False)

        logging.info("Setup python")
        python_dir = os.path.join(self.source_dir, "python")
        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            installer_name = self.get_conda_installer_name(self.os_name)
            installer_path = os.path.join(python_dir, installer_name)
            if self.os_name == consts.OS_Windows:
                cmd = "cmd /c start /wait " + installer_path + " /S /AddToPath=0 /RegisterPython=0 /D=" + self.python_dir
            elif self.os_name == consts.OS_Linux:
                cmd = "bash " + installer_path + " -b -p " + self.python_dir
            else:
                raise RuntimeError("Unknown OS: %s" % self.os_name)
            util.execute_cmd(cmd)

            conda_exe = util.locate_conda_exe(self.python_dir, self.os_name)
            channel_dir = os.path.join(self.source_dir, "packages", "channel",
                                       util.get_conda_channel_name(self.os_name, self.architecture))

            logging.info("Install conda-build")
            cmd = conda_exe + " install conda-build -y -q --offline --copy -c file:///" + channel_dir
            util.execute_cmd(cmd)
        else:
            if self.os_name == consts.OS_Windows:
                util.copy_directory(python_dir, self.python_dir)
                python_exe = os.path.join(self.python_dir, "python.exe")
            elif self.os_name == consts.OS_Linux:
                python_exe = "python" + ".".join(self.python_version.split('.')[0:2])
            else:
                raise RuntimeError("Unknown OS: %s" % self.os_name)

        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            logging.info("Create conda environment")
            cmd = conda_exe + " create -p " + self.env_dir + " -y -q --copy --offline python=" + self.python_version + " -c file:///" + channel_dir
            util.execute_cmd(cmd)
        else:
            logging.info("Create virtual environment")
            util.execute_cmd(python_exe + " -m venv " + self.env_dir)

        python_exe = util.locate_python3_exe(self.env_dir, self.os_name)
        if self.copy_site_packages:
            logging.info("Copy site packages")
            src = os.path.join(self.source_dir, "env")
            dest = os.path.join(self.env_dir)
            util.copy_directory(src, dest)
        else:
            if self.python_dist == consts.PYTHON_DIST_ANACONDA:
                logging.info("Install conda packages")
                cmd = conda_exe + " install -p " + self.env_dir + " -y -q --copy --offline --no-deps --file=" + self.conda_spec_path + " -c file:///" + channel_dir
                util.execute_cmd(cmd)

            logging.info("Upgrade pip")
            package_dir = os.path.join(self.source_dir, "packages")
            cmd = python_exe + " -m pip install pip --no-cache-dir --no-index --upgrade --force-reinstall --find-links=" + package_dir
            util.execute_cmd(cmd)
            util.execute_cmd(python_exe + " -m pip -V --disable-pip-version-check")

            logging.info("Install pip packages")
            cmd = python_exe + " -m pip install -U --force-reinstall -r " + self.requirements_path + " --no-index --no-cache-dir --no-deps --find-links=" + package_dir
            util.execute_cmd(cmd)

        util.execute_cmd(python_exe + " -m pip freeze --all --disable-pip-version-check")
        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            util.resolve_conda_package_conflict(self.python_dir, self.env_dir, self.required_conda_packages)

    def get_conda_installer_name(slef, os_name):
        if os_name == consts.OS_Windows:
            return "installer.exe"
        else:
            return "installer.sh"

    def get_arg_value(self, arg, default=None):
        return util.get_arg_value(self.args, arg, default)

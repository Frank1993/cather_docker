import logging
import os
import re
import tempfile

import repository
from shared import util, consts, fsutil


class PythonRuntimeBuilder:
    def __init__(self, args):
        self.args = args

    def run(self):
        self.init()
        self.build_all()
        return True

    def init(self):
        self.os_name = util.get_os_name()
        self.architectures = self.get_arg_value(consts.ARG_ARCHITECTURES, util.get_cpu_architecture()).split(',')

        self.data_root = util.get_data_root(self.args)
        self.source_dir = os.path.abspath(self.get_arg_value(consts.ARG_BUILD_SOURCE))

        self.local_package_dirs = []
        local_package_dir = os.path.join(self.source_dir, "packages")
        if os.path.isdir(local_package_dir):
            self.local_package_dirs.append(local_package_dir)

        local_package_dir = self.get_arg_value(consts.ARG_LOCAL_PACKAGE_DIR)
        if local_package_dir:
            self.local_package_dirs.append(os.path.abspath(local_package_dir))

        self.publish_dir = self.get_arg_value(consts.ARG_PUBLISH_DIR)
        if not self.publish_dir:
            self.publish_dir = os.path.join(self.source_dir, "publish")
        else:
            self.publish_dir = os.path.abspath(self.publish_dir)
        self.publish_dir = os.path.join(self.publish_dir, self.os_name)

        # Build may fail if default build directory path is long
        # On Windows, the root cause is path too long. On Linux, it fails due to bash shebang limitation for conda
        # Use temp directory usually can resolve this issue
        self.use_temp_build_dir = self.get_arg_value(consts.ARG_USE_TEMP_BUILD_DIR)
        if self.use_temp_build_dir:
            self.build_dir = tempfile.TemporaryDirectory().name
        else:
            self.build_dir = os.path.join(self.source_dir, "build")

        self.python_dir = os.path.join(self.build_dir, "python")
        self.env_dir = os.path.join(self.build_dir, "env")
        self.runtime_dir = os.path.join(self.build_dir, "runtime")
        self.package_dir = os.path.join(self.runtime_dir, "packages")
        self.reserve_build_dir = self.get_arg_value(consts.ARG_RESERVE_BUILD_DIR)
        if self.use_temp_build_dir:
            self.reserve_build_dir = False
        self.copy_site_packages = self.get_arg_value(consts.ARG_COPY_SITE_PACKAGES)

        config_path = os.path.join(self.source_dir, consts.RUNTIME_INI)
        config = util.parse_ini(config_path)

        section = config["Python"]
        self.python_version = section.get("Version")
        self.python_dist = section.get("Distribution", consts.PYTHON_DIST_CPYTHON)
        self.python_dist_version = section.get("DistributionVersion", self.python_version)
        self.local_packages = [name.strip() for name in section.get("LocalPackages", "").split(',') if name]
        self.requirements_path = os.path.join(self.runtime_dir, section.get("Requirements", consts.REQUIREMENTS_TXT))

        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            self.conda_spec_path = os.path.join(self.runtime_dir, section.get("CondaSpec", consts.CONDA_SPEC_TXT))
            self.required_conda_packages = [name.strip() for name in section.get("RequiredCondaPackages", "").split(',')
                                            if name]
            if self.copy_site_packages:
                raise RuntimeError("Copy site packages does not support anaconda")

    def build_all(self):
        util.delete_directory(self.publish_dir)

        for architecture in self.architectures:
            logging.info("Build runtime for %s %s", self.os_name, architecture)
            self.build(architecture)
            logging.info("Build runtime succeed")

    def build(self, architecture):
        logging.info("Copy source to build directory")
        util.recreate_directory(self.build_dir)
        util.copy_directory(self.source_dir, self.runtime_dir, False)
        for dir_name in next(os.walk(self.source_dir))[1]:
            if dir_name not in ["packages", "build", "publish"]:
                util.copy_directory(os.path.join(self.source_dir, dir_name), os.path.join(self.runtime_dir, dir_name))

        self.setup_python(architecture)
        self.setup_packages()
        self.publish_runtime(architecture)

        if not self.reserve_build_dir:
            util.delete_directory(self.build_dir)

    def setup_python(self, architecture):
        logging.info("Setup python: Version=%s, Distribution=%s, DistributionVersion=%s", self.python_version,
                     self.python_dist, self.python_dist_version)

        if self.python_dist == consts.PYTHON_DIST_CPYTHON:
            self.setup_cpython(architecture)
        elif self.python_dist == consts.PYTHON_DIST_ANACONDA:
            self.setup_conda(architecture)
        else:
            raise RuntimeError("Unsupported python distribution: %s" % self.python_dist)

    def setup_cpython(self, architecture):
        # For Windows, python installation is portable. It does not require that system installed python matches build requirements.
        # For Linux, didn't find a way to make python portable, have to use system installed python. The python version must match build requirements.
        if self.os_name == consts.OS_Windows:
            repo = repository.get_repository("Python", self.data_root)
            python_src_dir = repo.get_item_directory(self.python_dist, self.python_dist_version, self.os_name,
                                                     architecture)

            logging.info("Copy python from %s", python_src_dir)
            fsutil.copy_directory(python_src_dir, self.build_dir)
            util.unzip(os.path.join(self.build_dir, "python.zip"), self.python_dir)

            python_exe = os.path.join(self.python_dir, "python.exe")
        elif self.os_name == consts.OS_Linux:
            logging.info("Use system python")
            # Use full python exe name, e.g. python3.6, 'which' cmd will fail if the specific python does not exist
            python_exe = "python" + ".".join(self.python_version.split('.')[0:2])
            util.execute_cmd("which " + python_exe)
        else:
            raise RuntimeError("Unknown OS: %s" % self.os_name)

        # Create a clean venv
        logging.info("Create virtual environment")
        util.execute_cmd(python_exe + " -m venv " + self.env_dir)

    def setup_conda(self, architecture):
        # Install miniconda which is small, https://conda.io/miniconda.html
        repo = repository.get_repository("Python", self.data_root)
        python_src_dir = repo.get_item_directory(self.python_dist, self.python_dist_version, self.os_name, architecture)

        logging.info("Copy python from %s", python_src_dir)
        fsutil.copy_directory(python_src_dir, self.build_dir)

        # Install in silent mode, https://conda.io/docs/user-guide/install/index.html
        installer_path = os.path.join(self.build_dir, self.get_conda_installer_name(self.os_name))
        if self.os_name == consts.OS_Windows:
            cmd = 'cmd /c start /wait %s /S /AddToPath=0 /RegisterPython=0 /D=%s' % (installer_path, self.python_dir)
        elif self.os_name == consts.OS_Linux:
            cmd = 'bash %s -b -p %s' % (installer_path, self.python_dir)
        else:
            raise RuntimeError("Unknown OS: %s" % self.os_name)

        logging.info("Install python")
        util.execute_cmd(cmd)
        conda_exe = util.locate_conda_exe(self.python_dir, self.os_name)

        # Need to install conda-build for creating custom channel
        logging.info("Install conda-build")
        util.execute_cmd(conda_exe + " install conda-build -y -q --copy")

        # Create a clean env with give python version
        # Use copy instead of hard link here, because some pip packages may modify its directory under site-package folder
        # If you install a conda package first, and then install its pip package, the original conda package file may get corrupted if using hard link
        logging.info("Create conda environment")
        util.execute_cmd("%s create -p %s -y -q --copy python=%s" % (conda_exe, self.env_dir, self.python_version))

    def setup_packages(self):
        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            conda_exe = util.locate_conda_exe(self.python_dir, self.os_name)

            # If anaconda metapackage exists, install it seperately
            # Install anaconda metapackage with other packages (with version specified) may behavior unexpected
            anaconda_meta_spec = None
            conda_spec = []
            for line in [line for line in util.read_all_lines(self.conda_spec_path) if
                         line and not line.startswith('#')]:
                package = re.split(r'[<>=]+', line)[0]
                if package.lower() == "anaconda":
                    anaconda_meta_spec = line
                else:
                    conda_spec.append(line)

            if anaconda_meta_spec:
                logging.info("Install anaconda metapackage")
                util.execute_cmd(conda_exe + " install -p " + self.env_dir + " --copy -y -q " + anaconda_meta_spec)
                # Need to remove anaconda metapackage, otherwise it may become anaconda-custom after other packages were installed
                util.execute_cmd(conda_exe + " remove -p " + self.env_dir + " -y anaconda")
                util.write_all_lines(self.conda_spec_path, conda_spec)

            logging.info("Install conda packages")
            cmd = conda_exe + " install -p " + self.env_dir + " --copy -y -q --file " + self.conda_spec_path
            util.execute_cmd(cmd)
            util.execute_cmd(conda_exe + " list -p " + self.env_dir)

        logging.info("Setup pip packages")
        python_exe = util.locate_python3_exe(self.env_dir, self.os_name)
        util.create_directory(self.package_dir)

        # Upgrade pip first, we need some new pip options and old pip cannot install tensorflow
        # Legacy pip does not have -download command, has to use -d
        logging.info("Upgrade pip")
        cmd = python_exe + " -m pip install pip --no-cache-dir -d " + self.package_dir
        util.execute_cmd(cmd)
        util.execute_cmd(python_exe + " -m pip install pip --upgrade --force-reinstall --no-cache-dir --no-index --find-links=" + self.package_dir)
        util.execute_cmd(python_exe + " -m pip -V --disable-pip-version-check")

        if len(self.local_package_dirs) > 0:
            logging.info("Download packages from local")
            cmd = python_exe + " -m pip download %s --no-cache-dir --no-index --no-deps -d " + self.package_dir
            for local_dir in self.local_package_dirs:
                cmd += " --find-links=" + local_dir

            requirements = [line for line in util.read_all_text(self.requirements_path).splitlines() if
                            line and not line.startswith('#')]
            for requirement in requirements:
                util.execute_cmd(cmd % requirement, check_exit_code=False)

            logging.info("Packages downloaded from local:")
            local_packages = {}
            for file_name in next(os.walk(self.package_dir))[2]:
                logging.info("\t%s", file_name)
                package = file_name.split('-')[0]
                local_packages[util.normalize_python_package_name(package)] = os.path.join(self.package_dir, file_name)

            logging.info("Check local package override")
            for package in self.local_packages:
                if util.normalize_python_package_name(package) not in local_packages:
                    raise RuntimeError("Local required package %s is missing" % package)

            logging.info("Update requirements.txt for local packages")
            for i in range(len(requirements)):
                package = re.split(r'[<>=]+', requirements[i])[0]
                if util.normalize_python_package_name(package) in local_packages:
                    requirement = local_packages[util.normalize_python_package_name(package)]
                    logging.info("\tReplace %s with %s", requirements[i], requirement)
                    requirements[i] = requirement

            util.write_all_text(self.requirements_path, os.linesep.join(requirements))

        # Download packages from pypi
        logging.info("Download packages from pypi")
        cmd = python_exe + " -m pip download --no-cache-dir -d " + self.package_dir + " -r " + self.requirements_path + " --find-links=" + self.package_dir
        util.execute_cmd(cmd)

        # Install packages, we prefer pip package over conda package
        logging.info("Install pip packages")
        util.execute_cmd(python_exe + " -m pip freeze --all")
        cmd = python_exe + " -m pip install -r " + self.requirements_path + " --no-index --no-cache-dir -U --force-reinstall --find-links=" + self.package_dir
        util.execute_cmd(cmd)

        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            installed_conda_packages = util.resolve_conda_package_conflict(self.python_dir, self.env_dir,
                                                                           self.required_conda_packages)

            logging.info("Export conda_spec.txt")
            _, conda_output = util.execute_cmd(conda_exe + " list -e -p " + self.env_dir)
            util.write_all_text(self.conda_spec_path, conda_output)

            _, pip_output = util.execute_cmd(python_exe + " -m pip freeze --all")
            requirements = []
            for requirement in [line for line in pip_output.splitlines() if line and not line.startswith('#')]:
                package = util.normalize_python_package_name(re.split(r'[<>=]+', requirement)[0])
                # Remove conda packages from requirements
                if package not in installed_conda_packages:
                    requirements.append(requirement)

            logging.info("Export requirements.txt:\n%s", '\n'.join(requirements))
            util.write_all_lines(self.requirements_path, requirements)
        else:
            logging.info("Export requirements.txt")
            _, pip_output = util.execute_cmd(python_exe + " -m pip freeze --all")
            util.write_all_text(self.requirements_path, pip_output)

    def get_conda_installer_name(slef, os_name):
        if os_name == consts.OS_Windows:
            return "installer.exe"
        else:
            return "installer.sh"

    def publish_runtime(self, architecture):
        python_dir = os.path.join(self.runtime_dir, "python")
        if self.python_dist == consts.PYTHON_DIST_ANACONDA:
            installer_name = self.get_conda_installer_name(self.os_name)
            installer_path = os.path.join(self.build_dir, installer_name)
            util.copy_file(installer_path, os.path.join(python_dir, installer_name))

            logging.info("Build conda channel")
            conda_exe = util.locate_conda_exe(self.python_dir, self.os_name)
            channel_dir = os.path.join(self.package_dir, "channel",
                                       util.get_conda_channel_name(self.os_name, architecture))

            pkgs_dir = os.path.join(self.python_dir, "pkgs")
            for package in next(os.walk(pkgs_dir))[2]:
                if package.endswith(".bz2"):
                    util.copy_file(os.path.join(pkgs_dir, package), os.path.join(channel_dir, package))

            util.execute_cmd(conda_exe + " index " + channel_dir)
        else:
            if self.os_name == consts.OS_Windows:
                util.unzip(os.path.join(self.build_dir, "python.zip"), python_dir)

            if self.copy_site_packages:
                logging.info("Copy site packages")
                env_dir = os.path.join(self.runtime_dir, "env")
                src = util.locate_site_packages_dir(self.env_dir, self.os_name, self.python_version)
                dest = util.locate_site_packages_dir(env_dir, self.os_name, self.python_version)
                util.copy_directory(src, dest)
                util.delete_directory(self.package_dir)

        logging.info("Update runtime.ini")
        config_path = os.path.join(self.runtime_dir, consts.RUNTIME_INI)
        config = util.parse_ini(config_path)

        section = config["Runtime"]
        section["OS"] = self.os_name
        section["Architecture"] = architecture

        section = config["Python"]
        if self.copy_site_packages:
            section["CopySitePackages"] = str(self.copy_site_packages)

        with open(config_path, 'w') as configfile:
            config.write(configfile)

        logging.info("Publish runtime")
        util.clean_pycache(self.runtime_dir)
        util.zip_directory(self.runtime_dir, os.path.join(self.publish_dir, architecture, consts.RUNTIME_ZIP_NAME))

    def get_arg_value(self, arg, default=None):
        return util.get_arg_value(self.args, arg, default)

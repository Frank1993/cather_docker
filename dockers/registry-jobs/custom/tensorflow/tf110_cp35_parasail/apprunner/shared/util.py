import configparser
import itertools
import json
import logging
import math
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

from shared import consts


def abort(msg=None, *args, status=1):
    if msg is not None:
        logging.error(msg % args)
    sys.exit(status)


def print_argv(args=sys.argv):
    logging.info("=============================================")
    args_str = ""
    for arg in args:
        if any([c in ["'", '"', ' '] for c in arg]):
            args_str += ('"' + arg + '" ')
        else:
            args_str += (arg + ' ')
    logging.info(args_str)


def print_cmd_args(args):
    logging.info("AppRunner %s arguments:", args.subparser_name)
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None and arg not in ["func", "subparser_name"]:
            logging.info("\t%s=%s", arg, getattr(args, arg))


def get_arg_attribute(arg: str):
    return arg.lstrip('-').replace('-', '_')


def convert_arg_dash_to_underscore(arg: str):
    name = arg.lstrip('-').replace('-', '_')
    return '-' * (len(arg) - len(name)) + name


def get_arg_value(args, arg, default=None):
    attr_name = get_arg_attribute(arg)
    if not hasattr(args, attr_name):
        return default

    value = getattr(args, attr_name)
    if value is not None:
        return value
    else:
        return default


def set_arg_value(args, arg, value):
    setattr(args, get_arg_attribute(arg), value)


def get_normalized_argv(argv):
    args_str = ""
    for arg in argv:
        if arg is not None:
            args_str += '"' + str(arg) + '" '
    return args_str


def create_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)


def delete_directory(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)


def recreate_directory(dir):
    delete_directory(dir)
    create_directory(dir)


def get_working_root(args):
    dir = get_arg_value(args, consts.ARG_APPRUNNER_WORKING_ROOT)
    if dir:
        return os.path.abspath(dir)
    else:
        return os.path.join(os.getcwd(), ".working")


def get_data_root(args):
    dir = get_arg_value(args, consts.ARG_APPRUNNER_DATA_ROOT, consts.APPRUNNER_COSMOS_ROOT)
    if is_cosmos_path(dir):
        return dir

    return os.path.abspath(dir)


def copy_file(src, dest):
    dir = os.path.dirname(dest)
    create_directory(dir)
    shutil.copyfile(src, dest)


def copy_directory(src, dest, recursive=True):
    create_directory(dest)

    logging.info("Read permission is %s", os.access(src, os.R_OK))
    logging.info("Write permission is %s", os.access(src, os.W_OK))
    logging.info("Execute permission is %s", os.access(src, os.X_OK))

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dest, item)
        if os.path.isfile(s):
            shutil.copyfile(s, d)
        elif recursive:
            copy_directory(s, d)


def get_os_name():
    return platform.system()


# http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
def get_cpu_architecture():
    if sys.maxsize > 2 ** 32:
        return "x64"
    else:
        return "x86"


def get_host_name():
    return platform.node()


# http://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-of-eth0-in-python
def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def get_host_ib_ip(ifname="ib0"):
    if get_os_name() != consts.OS_Linux:
        raise RuntimeError("get_host_ib_ip currently only support linux")
    return os.popen('ip addr show ' + ifname).read().split("inet ")[1].split("/")[0]


def get_current_python_path():
    return sys.executable


def get_python_version_info():
    return sys.version_info


def get_environment_variable(name, default=None):
    if name not in os.environ:
        return default
    return os.environ[name]


def is_cosmos_path(url: str):
    if url is None:
        return False

    return url.lower().startswith("https://cosmos")


def is_hdfs_path(url: str):
    if url is None:
        return False

    return url.lower().startswith("hdfs://")


def read_all_text(path):
    with open(path) as f:
        return f.read()


def read_all_lines(path):
    return read_all_text(path).splitlines()


def write_all_text(path, content=""):
    with open(path, 'w') as file:
        file.write(content)


def write_all_lines(path, lines):
    write_all_text(path, '\n'.join(lines))


def apend_all_text(path, content=""):
    with open(path, 'a') as file:
        file.write(content)


def append_all_lines(path, lines):
    apend_all_text(path, '\n'.join(lines))


def parse_ini(path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)
    return config


def load_ini_as_dict(content):
    mapping = dict()

    for line in content.splitlines():
        if line.startswith('#'):
            continue

        (k, v) = [x.strip() for x in line.split('=', 1)]
        if k is not None:
            mapping[k] = v

    return mapping


def zip_directory(dir, archive):
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_base = os.path.join(tmp_dir, "tmp")
        zip_path = zip_base + ".zip"
        shutil.make_archive(zip_base, 'zip', dir)

        create_directory(os.path.dirname(archive))
        shutil.move(zip_path, archive)


def unzip(src, dest):
    create_directory(dest)
    with zipfile.ZipFile(src, 'r') as zipf:
        zipf.extractall(dest)


# https://docs.python.org/3/library/itertools.html
def roundrobin(*iterables):
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def wait_file(path, interval: int = 1, timeout: int = 1800):
    loop = math.ceil(timeout / interval)
    while loop > 0:
        if os.path.isfile(path):
            return True
        time.sleep(interval)
        loop -= 1
    return False


def get_mpi_int(name):
    value = get_environment_variable(name)
    if value is None:
        return value
    return int(value)


def get_mpi_word_size():
    return get_mpi_int("OMPI_COMM_WORLD_SIZE")


def get_mpi_local_size():
    return get_mpi_int("OMPI_COMM_WORLD_LOCAL_SIZE")


def get_mpi_local_rank():
    return get_mpi_int("OMPI_COMM_WORLD_LOCAL_RANK")


def execute_cmd(cmd, print=True, check_exit_code=True):
    if print:
        logging.info("Execute: %s", cmd)

    args = shlex.split(cmd, posix=False)
    process = subprocess.Popen(args, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        if print:
            sys.stdout.write(line)
            sys.stdout.flush()

    process.stdout.close()
    exit_code = process.wait()

    if check_exit_code and exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, cmd)

    return exit_code, output


def locate_python3_exe(python_dir, os_name):
    if os_name == consts.OS_Windows:
        # For actual python installation, python.exe is under root directory
        # For venv, python.exe is under Scripts directory
        python_exe = os.path.join(python_dir, "python.exe")
        if not os.path.isfile(python_exe):
            python_exe = os.path.join(python_dir, "Scripts", "python.exe")
    elif os_name == consts.OS_Linux:
        python_exe = os.path.join(python_dir, "bin", "python3")
    else:
        raise RuntimeError("Unknown OS: %s" % os_name)
    return python_exe


def locate_conda_exe(conda_dir, os_name):
    if os_name == consts.OS_Windows:
        conda_exe = os.path.join(conda_dir, "Scripts", "conda.exe")
    elif os_name == consts.OS_Linux:
        conda_exe = os.path.join(conda_dir, "bin", "conda")
    else:
        raise RuntimeError("Unknown OS: %s" % os_name)
    return conda_exe


def get_conda_channel_name(os_name, architecture):
    if os_name == consts.OS_Windows:
        name = "win"
    elif os_name == consts.OS_Linux:
        name = "linux"
    else:
        raise RuntimeError("Unsupported OS: %s" % os_name)

    name += "-"
    if architecture == "x64":
        name += "64"
    elif architecture == "x86":
        name += "32"
    else:
        raise RuntimeError("Unsupported CPU architecture: %s" % architecture)

    return name


def locate_site_packages_dir(python_dir, os_name, python_version):
    if os_name == consts.OS_Windows:
        path = os.path.join(python_dir, "Lib", "site-packages")
    elif os_name == consts.OS_Linux:
        path = os.path.join(python_dir, "lib", "python" + python_version, "site-packages")
    else:
        raise RuntimeError("Unknown OS: %s" % os_name)
    return path


# Resolve package conflict between conda and pip, return normalized conda installed package names from pip freeze perspective
def resolve_conda_package_conflict(conda_dir, env_dir, required_conda_packages=None):
    class CondaPackageInfo:
        pass

    logging.info("Check package conflicts")

    required_conda_packages = set(required_conda_packages) if required_conda_packages else set()
    os_name = get_os_name()
    conda_exe = locate_conda_exe(conda_dir, os_name)
    python_exe = locate_python3_exe(env_dir, os_name)
    python_version = '.'.join(execute_cmd(python_exe + " -V")[1].split(' ')[1].split('.')[0:2])
    meta_dir = os.path.join(env_dir, "conda-meta")
    site_packages_dir = locate_site_packages_dir(env_dir, os_name, python_version)

    # conda list -p shows all installed packages, either installed by pip or conda
    _, output = execute_cmd(conda_exe + " list -p " + env_dir)
    lines = [line for line in output.splitlines() if line and not line.startswith('#')]

    # Get packages installed by pip with normalized name
    pip_installed_packages = set()
    for line in lines:
        tokens = re.split(r'[ ]+', line)
        if tokens[2] == "<pip>":
            pip_installed_packages.add(normalize_python_package_name(tokens[0]))

    # Get packages installed by conda, key is normalized package name shows in pip freeze
    conda_installed_packages = dict()
    for line in lines:
        tokens = re.split(r'[ ]+', line)
        if tokens[2] == "<pip>":
            continue

        info = CondaPackageInfo()
        info.name = tokens[0]
        info.version = tokens[1]
        info.type = tokens[2]

        # Conda installed package may show a different name in pip freeze, try to find it
        name = info.name
        if name not in pip_installed_packages:
            meta_path = os.path.join(meta_dir, info.name + '-' + info.version + '-' + info.type + ".json")
            with open(meta_path) as file:
                data = json.load(file)
                for f in data["files"]:
                    m = re.search(r"site-packages/(.+)-" + info.version + ".*.egg.*", f)
                    if m:
                        name = m.group(1)
                        if name != info.name:
                            logging.info("Found conda/pip package name change, %s => %s", info.name, name)
                        break

        conda_installed_packages[normalize_python_package_name(name)] = info

    missing_packages = []
    conda_package_names = set(package.name for package in conda_installed_packages.values())
    for package in required_conda_packages:
        if package not in conda_package_names:
            missing_packages.append(package)

    if len(missing_packages) > 0:
        raise RuntimeError("Required conda package missing: %s" % ",".join(missing_packages))

    conda_package_names = set(conda_installed_packages.keys())
    conflict_packages = set.intersection(pip_installed_packages, conda_package_names)

    if len(conflict_packages) == 0:
        logging.info("No package conflict found")
        return conda_package_names

    logging.info("Found %s conflicts: %s", len(conflict_packages), ",".join(conflict_packages))
    logging.info("Resolve package conflicts")

    # We prefer pip package over conda package, unless a conda package is explicitly required
    reinstall_packages = []
    for package in conflict_packages:
        conda_info = conda_installed_packages.get(package)
        if conda_info and conda_info.name in required_conda_packages:
            reinstall_packages.append(package)
        else:
            conda_package_names.remove(package)

    # Uninstall all conflicts conda packages since these packages will either be removed or re-installed later
    # Remove conda-meta so they won't show in conda list
    # No need to remove egg info since we are using pip -U --force-reinstall instead of pip -I, so previous conda package will be uninstalled
    packages = " ".join(conflict_packages)
    logging.info("Uninstall conda packages: %s", packages)
    for package in conflict_packages:
        info = conda_installed_packages[package]
        path = os.path.join(meta_dir, info.name + '-' + info.version + '-' + info.type + ".json")
        logging.info("Delete %s", path)
        delete_file(path)

    # Reinstall conda package to overwrite its pip version, without dependency
    if len(reinstall_packages) > 0:
        # Uninstall pip package so it won't show as a pip package when calling conda-list
        packages = " ".join(reinstall_packages)
        logging.info("Uninstall pip packages: %s", packages)
        cmd = python_exe + " -m pip uninstall -y --disable-pip-version-check " + packages
        execute_cmd(cmd)

        # Re-install conda package with version specified. Otherwise, the latest version may always be prefered when there are multiple versions
        packages = " ".join([info.name + '=' + info.version + '=' + info.type for info in
                             [conda_installed_packages[package] for package in reinstall_packages]])
        logging.info("Reinstall conda packages: %s", packages)
        cmd = conda_exe + " install -y --copy --force --offline --no-update-deps -q -p " + env_dir + " " + packages
        execute_cmd(cmd)

    execute_cmd(conda_exe + " list -p " + env_dir)
    execute_cmd(python_exe + " -m pip freeze --all --disable-pip-version-check")
    return conda_package_names


# Parse "pip freeze" or "conda list -e" output
def parse_python_package_spec(spec):
    packages = []
    for line in [line for line in spec.splitlines() if line and not line.startswith('#')]:
        package = re.split(r'[<>=]+', line)[0]
        packages.append(normalize_python_package_name(package))
    return packages


# https://www.python.org/dev/peps/pep-0426/#name
def normalize_python_package_name(name):
    return name.replace('-', '_').lower()


def clean_pycache(dir: str):
    for item in os.listdir(dir):
        path = os.path.join(dir, item)
        if os.path.isdir(path):
            if item == "__pycache__":
                delete_directory(path)
            else:
                clean_pycache(path)


def try_parse_int(value: str, default: int):
    try:
        return int(value)
    except ValueError:
        return default

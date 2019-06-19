import argparse
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree
import zipfile

APPRUNNER_COSMOS_ROOT = "https://cosmos08.osdinfra.net/cosmos/cosmos.upload/shares/bingads.data.clickpredict/data/ClickPrediction/TrainStation/AppRunner/"
VERSIONS_INI = "versions.ini"


def parse_source_dir(args):
    dir = args.apprunner_source_dir
    if dir is None:
        return build_source_dir(args)

    if is_cosmos_path(dir):
        return dir

    return os.path.abspath(dir)


def build_source_dir(args):
    root = args.apprunner_data_root
    if root is None:
        root = APPRUNNER_COSMOS_ROOT

    is_cosmos = is_cosmos_path(root)
    if is_cosmos:
        dir = root + "Scripts/apprunner/"
    else:
        dir = os.path.join(os.path.abspath(root), "Scripts", "apprunner")

    version = args.apprunner_version
    if version is None:
        ini = os.path.join(dir, VERSIONS_INI)
        logging.info("Parse apprunner latest version from %s", ini)
        version = get_latest_version(ini)
        if version is None:
            logging.error("Failed to get latest version")
            return None

    logging.info("Apprunner version is %s", version)

    if is_cosmos:
        dir = dir + version + "/"
    else:
        dir = os.path.join(dir, version)

    return dir


def get_latest_version(path):
    versions = dict()

    if is_cosmos_path(path):
        content = cosmos_read_all_text(path)
    else:
        content = real_all_text(path)

    for line in content.splitlines():
        if line.startswith('#'):
            continue

        (k, v) = [x.strip() for x in line.split('=', 1)]
        if k is not None:
            versions[k] = v

    return versions.get("latest", None)


def create_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)


def copy_directory(src, dest):
    create_directory(dest)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dest, item)
        if os.path.isfile(s):
            shutil.copyfile(s, d)
        else:
            copy_directory(s, d)


def real_all_text(path):
    with open(path) as f:
        return f.read()


def is_cosmos_path(url: str):
    if url is None:
        return False
    return url.lower().startswith("https://cosmos")


def cosmos_stream_exists(stream):
    try:
        with urllib.request.urlopen(stream + "?property=info"):
            return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        else:
            raise err


def cosmos_read_all_text(stream, encoding="utf-8"):
    with urllib.request.urlopen(stream) as response:
        return response.read().decode(encoding)


def cosmos_download_file(src, dest, overwrite=True):
    dest = os.path.abspath(dest)
    create_directory(os.path.dirname(dest))

    if overwrite and os.path.isfile(dest):
        os.remove(dest)

    urllib.request.urlretrieve(src, dest)


def cosmos_download_directory(src, dest, recursive=True, overwrite=True):
    dest = os.path.abspath(dest)
    create_directory(os.path.dirname(dest))

    response = cosmos_read_all_text(src + "?view=xml")
    root = xml.etree.ElementTree.fromstring(response)

    for info in root:
        stream = info.find("StreamName").text
        is_dir = info.find("IsDirectory").text
        if is_dir == "false":
            name = os.path.basename(stream)
            cosmos_download_file(stream, os.path.join(dest, name), overwrite)
        elif recursive:
            name = os.path.basename(os.path.dirname(stream))
            cosmos_download_directory(stream, os.path.join(dest, name), True, overwrite)


def print_arguments(args):
    args_str = ""
    for arg in args:
        if any([c in ["'", '"', ' '] for c in arg]):
            args_str += ('"' + arg + '" ')
        else:
            args_str += (arg + ' ')
    logging.info(args_str)


def get_normalized_argv(argv):
    args_str = ""
    for arg in argv:
        if arg is not None:
            args_str += '"' + str(arg) + '" '
    return args_str


def abort(msg=None, status=1):
    if msg is not None:
        logging.error(msg)

    sys.exit(status)


class ArgumentParser(argparse.ArgumentParser):
    def _get_option_tuples(self, option_string):
        result = []

        # option strings starting with two prefix characters are only
        # split at the '='
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if '=' in option_string:
                option_prefix, explicit_arg = option_string.split('=', 1)
            else:
                option_prefix = option_string
                explicit_arg = None
            for option_string in self._option_string_actions:
                if option_string == option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, explicit_arg
                    result.append(tup)

        # single character options can be concatenated with their arguments
        # but multiple character options always have to have their argument
        # separate
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix = option_string
            explicit_arg = None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]

            for option_string in self._option_string_actions:
                if option_string == short_option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, short_explicit_arg
                    result.append(tup)
                elif option_string == option_prefix:
                    action = self._option_string_actions[option_string]
                    tup = action, option_string, explicit_arg
                    result.append(tup)

        # shouldn't ever get here
        else:
            self.error(_('unexpected option string: %s') % option_string)

        # return the collected option tuples
        return result


logging.basicConfig(stream=sys.stdout, format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.INFO)

argv = sys.argv[1:]
parser = ArgumentParser()
parser.add_argument("--apprunner-version")
parser.add_argument("--apprunner-source-dir")
parser.add_argument("--apprunner-data-root")

print_arguments([sys.argv[0]] + argv)
(args, pass_through) = parser.parse_known_args(argv)

working_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".working")
logging.info("Working root is %s", working_root)
if os.path.isdir(working_root):
    shutil.rmtree(working_root)

src_dir = parse_source_dir(args)
if src_dir is None:
    abort("Failed to parse apprunner source dir")

dest_dir = os.path.join(working_root, "apprunner")
create_directory(dest_dir)

use_zip = False
if is_cosmos_path(src_dir):
    zip_path = src_dir + "apprunner.zip"
    if (cosmos_stream_exists(zip_path)):
        use_zip = True
else:
    zip_path = os.path.join(src_dir, "apprunner.zip")
    if os.path.isfile(zip_path):
        use_zip = True

if use_zip:
    dest_path = os.path.join(dest_dir, "apprunner.zip")
    logging.info("Copy apprunner from %s", zip_path)
    if is_cosmos_path(src_dir):
        cosmos_download_file(zip_path, dest_path)
    else:
        shutil.copyfile(zip_path, dest_path)

    logging.info("Unzip %s", dest_path)
    with zipfile.ZipFile(dest_path, 'r') as zipf:
        zipf.extractall(dest_dir)

    os.remove(dest_path)
else:
    logging.info("Copy apprunner from %s", src_dir)
    if is_cosmos_path(src_dir):
        cosmos_download_directory(src_dir, dest_dir)
    else:
        copy_directory(src_dir, dest_dir)

run_args = "run " + get_normalized_argv(argv + ["--apprunner-working-root", working_root])
cmd = "{} {} {}".format(sys.executable, os.path.join(dest_dir, "apprunner.py"), run_args)

logging.info("Start apprunner: %s", cmd)
status = os.system(cmd)
if status == 0:
    logging.info("Apprunner succeed")
else:
    logging.error("Apprunner failed, exit_code=%s", status)
    abort()

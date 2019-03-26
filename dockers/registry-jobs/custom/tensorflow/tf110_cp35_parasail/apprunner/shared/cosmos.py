import os
import urllib.error
import urllib.request
import xml.etree.ElementTree

from shared import util


def parse_cosmos_path_file(path):
    if path is None:
        return None

    if not os.path.isfile(path):
        return None

    lines = util.read_all_text(path).splitlines()
    if len(lines) < 2:
        return None

    vc_path = lines[1].strip() + lines[0].strip().strip('"').strip()
    return vc_to_http_path(vc_path)


def vc_to_http_path(vc_path: str):
    if vc_path is None:
        return None

    prefix = "vc://"
    if not vc_path.startswith(prefix):
        return None

    strs = vc_path[len(prefix):].split('/', 2)
    cluster = strs[0]
    vc = strs[1]
    path = strs[2]

    return "https://{}.osdinfra.net/cosmos/{}/{}".format(cluster, vc, path)


def read_all_text(stream, encoding="utf-8"):
    with urllib.request.urlopen(stream) as response:
        return response.read().decode(encoding)


def stream_exists(stream):
    try:
        with urllib.request.urlopen(stream + "?property=info"):
            return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        else:
            raise err


def directory_exists(dir):
    try:
        with urllib.request.urlopen(dir + "?view=xml"):
            return True
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        else:
            raise err


def download_file(src, dest, overwrite=True):
    dest = os.path.abspath(dest)
    util.create_directory(os.path.dirname(dest))

    if overwrite and os.path.isfile(dest):
        os.remove(dest)

    urllib.request.urlretrieve(src, dest)


def download_directory(src, dest, recursive=True, overwrite=True):
    dest = os.path.abspath(dest)
    util.create_directory(os.path.dirname(dest))

    response = read_all_text(src + "?view=xml")
    root = xml.etree.ElementTree.fromstring(response)

    for info in root:
        stream = info.find("StreamName").text
        is_dir = info.find("IsDirectory").text
        if is_dir == "false":
            name = os.path.basename(stream)
            download_file(stream, os.path.join(dest, name), overwrite)
        elif recursive:
            name = os.path.basename(os.path.dirname(stream))
            download_directory(stream, os.path.join(dest, name), True, overwrite)

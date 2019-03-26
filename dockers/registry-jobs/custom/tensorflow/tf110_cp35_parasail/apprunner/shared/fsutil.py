import os

from shared import util, cosmos


def abspath(path):
    if util.is_cosmos_path(path):
        return path
    else:
        return os.path.abspath(path)


def dir_join_path(path, *paths):
    if not util.is_cosmos_path(path):
        return os.path.abspath(os.path.join(path, *paths))

    for name in paths:
        path = path + name + "/"

    return path


def file_join_path(path, *paths):
    if not util.is_cosmos_path(path):
        return os.path.abspath(os.path.join(path, *paths))

    for name in paths:
        path = path + name + "/"

    return path[:len(path) - 1]


def is_file(path):
    if not util.is_cosmos_path(path):
        return os.path.isfile(path)
    else:
        return not path.endswith('/')


def file_exists(path):
    if not util.is_cosmos_path(path):
        return os.path.isfile(path)
    else:
        return cosmos.stream_exists(path)


def directory_exists(dir):
    if not util.is_cosmos_path(dir):
        return os.path.isdir(dir)
    else:
        return cosmos.directory_exists(dir)


def copy_file(src, dest):
    if not util.is_cosmos_path(src):
        util.copy_file(src, dest)
    else:
        cosmos.download_file(src, dest)


def copy_directory(src, dest):
    if not util.is_cosmos_path(src):
        util.copy_directory(src, dest)
    else:
        cosmos.download_directory(src, dest)


def read_all_text(path):
    if not util.is_cosmos_path(path):
        return util.read_all_text(path)
    else:
        return cosmos.read_all_text(path)

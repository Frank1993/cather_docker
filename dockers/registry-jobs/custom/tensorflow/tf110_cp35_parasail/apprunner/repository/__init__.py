from repository.os_arch_repository import OSArchRepository
from shared import consts, fsutil


def get_repository(name, data_root=None):
    if data_root is None:
        data_root = consts.APPRUNNER_COSMOS_ROOT
    else:
        data_root = fsutil.abspath(data_root)

    if name == "Python":
        return OSArchRepository(fsutil.dir_join_path(data_root, "Pythons"))
    elif name == "Runtime":
        return OSArchRepository(fsutil.dir_join_path(data_root, "Runtimes"))
    else:
        return None

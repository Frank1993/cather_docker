from shared import fsutil, consts, util


class Repository:
    def __init__(self, root):
        self.root = fsutil.abspath(root)

    def get_version(self, name, version=None):
        v = version
        if not v:
            v = consts.LATEST

        mapping = self.get_version_mapping(name, v)
        if not mapping:
            return version

        return mapping

    def get_version_mapping(self, name, version):
        if not version:
            return None

        path = fsutil.file_join_path(self.root, name, consts.VERSIONS_INI)
        if not fsutil.file_exists(path):
            return None

        content = fsutil.read_all_text(path)
        mappings = util.load_ini_as_dict(content)

        if version not in mappings:
            return None

        return mappings[version]

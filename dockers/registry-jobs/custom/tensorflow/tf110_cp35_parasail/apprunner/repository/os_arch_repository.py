from repository.repository import Repository
from shared import fsutil


class OSArchRepository(Repository):
    def get_item_directory(self, name, version, os_name, architecture):
        version = self.get_version(name, version)
        if version is None:
            return None

        return fsutil.dir_join_path(self.root, name, version, os_name, architecture)

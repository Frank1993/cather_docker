from configparser import ConfigParser

from trip_downloader.constants import *


class Config(object):
    def __init__(self, cfg_path):
        self.config = ConfigParser()
        self.config.read(cfg_path)
        self.sections = self.config.sections()

        self.db_user = self._get_option('db_credentials', 'username')
        self.db_password = self._get_option('db_credentials', 'password')
        self.db_host = self._get_option('db_credentials', 'host')
        self.db_port = self._get_option('db_credentials', 'port')
        self.db_name = self._get_option('db_credentials', 'db_name')

        self.osc_user = self._get_option('osc_credentials', 'username')
        self.osc_password = self._get_option('osc_credentials', 'password')

        self.signs = self._split_list(self._get_option('list', 'signs'))
        self.regions = self._split_list(self._get_option('list', 'regions'))
        self.trip_ids_included = self._split_list(self._get_option('list', 'trip_ids_included'))
        self.trip_ids_excluded = self._split_list(self._get_option('list', 'trip_ids_excluded'))

        self.manual = self._get_option('flag', 'manual') == YES
        self.automatic = self._get_option('flag', 'automatic') == YES
        self.confirmed = self._get_option('flag', 'confirmed') == YES
        self.removed = self._get_option('flag', 'removed') == YES
        self.to_be_checked = self._get_option('flag', 'to_be_checked') == YES
        self.full_trips = self._get_option('flag', 'full_trips') == YES
        self.sign_components = self._get_option('flag', 'sign_components') == YES
        self.telenav_user = self._get_option('flag', 'telenav_user') == YES
        self.proto_rois_only = self._get_option('flag', 'proto_rois_only') == YES
        self.remove_duplicates = self._get_option('flag', 'remove_duplicates') == YES

        self.min_signs_size = int(self._get_option('data', 'min_signs_size'))
        self.gte_timestamp = self._get_option('data', 'gte_timestamp')
        self.lte_timestamp = self._get_option('data', 'lte_timestamp')

    def _get_option(self, section, option):
        if section in self.sections:
            options = self.config.options(section)
            if option in options:
                value = self.config.get(section, option)
                return value

        return None

    @staticmethod
    def _split_list(list_content):
            return list_content.replace('\n', '').replace(' ', '').split(',')


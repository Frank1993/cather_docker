import logging
import os
import shutil

import apollo_python_common.log_util as log_util
from unittests.utils import resource_utils as resource_utils

class AbstractBaseTest(object):
    ftp_path = None

    def setUp(self):
        log_util.config(__file__)
        logger = logging.getLogger(__name__)
        logger.info("Downloading resources...")
        resource_utils.ensure_test_resource(self.ftp_path, self.resources_folder)
        
    def tearDown(self):
        logger = logging.getLogger(__name__)
        logger.info("Cleaning up resources...")
        shutil.rmtree(os.path.join(resource_utils.LOCAL_TEST_RESOURCES_FOLDER, self.resources_folder))

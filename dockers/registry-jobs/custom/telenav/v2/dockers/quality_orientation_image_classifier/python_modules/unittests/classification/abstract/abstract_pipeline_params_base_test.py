import os
import sys

from classification.scripts.constants import PipelineParamsBuilder

from unittests.common.abstract_base_test import AbstractBaseTest
from unittests.utils import resource_utils as resource_utils

sys.path.append(os.path.abspath('../../'))


class AbstractPipelineParamsBaseTest(AbstractBaseTest):

    def setUp(self):
        super(AbstractPipelineParamsBaseTest, self).setUp()
        self.params = PipelineParamsBuilder.build_params(resource_utils.LOCAL_TEST_RESOURCES_FOLDER + self.local_resources_path,
                                     nr_images_per_class=100,
                                     with_vp_crop=True,
                                     suffix="CONV_REV",
                                     img_dim=336)

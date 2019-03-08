import threading
import unittest
from time import sleep

import apollo_python_common.io_utils as io_utils
from mq_selectors.deploy.geo_image_selector.geo_selector import GeoSelector
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.mq.geo_selector.custom_mq_provider import CustomMQProvider

from tqdm import tqdm

class GeoSelectorTest(unittest.TestCase):

    def get_provider(self, output_queue, valid_region_2_images, invalid_region_2_images):
        conf = io_utils.config_load("../mq/configs/generic_config.json")
        conf["mq_output_queue_name"] = output_queue

        region_2_images = dict(list(valid_region_2_images.items()) + list(invalid_region_2_images.items()))
        return CustomMQProvider(conf, region_2_images)

    def get_accumulation_consumer(self, queue):
        conf = io_utils.json_load("../mq/configs/generic_config.json")
        conf["mq_input_queue_name"] = queue

        return AccumulationConsumer(conf)

    def get_geo_selector(self, input_queue, valid_region_2_images, output_queue):
        conf = io_utils.json_load("../mq/configs/generic_config.json")
        conf["mq_input_queue_name"] = input_queue
        conf["regions_to_process"] = valid_region_2_images
        conf["mq_output_queue_name"] = output_queue

        return GeoSelector(conf)

    def generic_test_selector(self, valid_region_2_images, invalid_region_2_images):
        input_queue = io_utils.get_random_file_name()
        output_queue = io_utils.get_random_file_name()

        provider = self.get_provider(input_queue, valid_region_2_images, invalid_region_2_images)
        selector = self.get_geo_selector(input_queue, valid_region_2_images, output_queue)

        valid_region_2_acc = {}
        for region in valid_region_2_images.keys():
            acc = self.get_accumulation_consumer(f"{region}_IMAGES")
            valid_region_2_acc[region] = acc
        output_acc = self.get_accumulation_consumer(output_queue)

        pt = threading.Thread(target=lambda: provider.start())
        pt.daemon = True
        pt.start()

        st = threading.Thread(target=lambda: selector.start())
        st.daemon = True
        st.start()

        for acc in tqdm(list(valid_region_2_acc.values())):
            at = threading.Thread(target=lambda: acc.start())
            at.daemon = True
            at.start()
        ot = threading.Thread(target=lambda: output_acc.start())
        ot.daemon = True
        ot.start()

        sleep(2)

        for region, acc in valid_region_2_acc.items():
            nr_expected_images = valid_region_2_images[region]
            nr_actual_images = len(acc.get_accumulated_protos())
            print("Region {} : {} = {}".format(region, nr_expected_images, nr_actual_images))
            assert (nr_expected_images == nr_actual_images)

        nr_expected_invalid_images = sum(invalid_region_2_images.values())
        nr_actual_invalid_images = len(output_acc.get_accumulated_protos())
        assert (nr_expected_invalid_images == nr_actual_invalid_images)
        print("Invalid : {} = {}".format(nr_expected_invalid_images, nr_actual_invalid_images))

        provider.delete_queue(input_queue)
        provider.delete_queue(output_queue)
        for region in valid_region_2_images.keys():
            provider.delete_queue(f"{region}_IMAGES")

    def test_selector_1(self):
        valid_region_2_images = {
            "US": 10,
            "EU": 5
        }
        invalid_region_2_images = {
            "RO": 20,
            "MX": 30
        }

        self.generic_test_selector(valid_region_2_images, invalid_region_2_images)

    def test_selector_2(self):
        valid_region_2_images = {
            "RO": 100,
        }
        invalid_region_2_images = {
            "MX": 10,
        }

        self.generic_test_selector(valid_region_2_images, invalid_region_2_images)

    def test_selector_3(self):
        valid_region_2_images = {
            "CH": 100,
            "JP": 1,
            "TZ": 55
        }
        invalid_region_2_images = {}

        self.generic_test_selector(valid_region_2_images, invalid_region_2_images)
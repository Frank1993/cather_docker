import threading
import unittest
from time import sleep

import apollo_python_common.io_utils as io_utils
from mq_selectors.deploy.quality_orientation_image_selector.quality_orientation_selector import QualityOrientationSelector
from unittests.mq.components.accumulation_consumer import AccumulationConsumer
from unittests.mq.quality_orientation_selector.custom_mq_provider import CustomMQProvider


class QualityOrientationSelectorTest(unittest.TestCase):

    def get_provider(self, output_queue, nr_good_protos, nr_bad_protos):
        conf = io_utils.config_load("../mq/configs/generic_config.json")
        conf["mq_output_queue_name"] = output_queue

        return CustomMQProvider(conf, nr_good_protos, nr_bad_protos)

    def get_accumulation_consumer(self, queue):
        conf = io_utils.json_load("../mq/configs/generic_config.json")
        conf["mq_input_queue_name"] = queue

        return AccumulationConsumer(conf)

    def get_quality_orientation_selector(self, input_queue, high_quality_queue_name, low_quality_queue_name):
        conf = io_utils.json_load("../mq/configs/generic_config.json")
        conf["mq_input_queue_name"] = input_queue
        conf["high_quality_queue_name"] = high_quality_queue_name
        conf["low_quality_queue_name"] = low_quality_queue_name

        return QualityOrientationSelector(conf)

    def generic_test_selector(self, nr_good_protos, nr_bad_protos):
        input_queue = io_utils.get_random_file_name()
        high_quality_queue_name = io_utils.get_random_file_name()
        low_quality_queue_name = io_utils.get_random_file_name()

        provider = self.get_provider(input_queue, nr_good_protos, nr_bad_protos)
        selector = self.get_quality_orientation_selector(input_queue, high_quality_queue_name, low_quality_queue_name)
        good_accumulator = self.get_accumulation_consumer(high_quality_queue_name)
        bad_accumulator = self.get_accumulation_consumer(low_quality_queue_name)

        provider_thread = threading.Thread(target=lambda: provider.start())
        provider_thread.daemon = True

        selector_thread = threading.Thread(target=lambda: selector.start())
        selector_thread.daemon = True

        good_accumulator_thread = threading.Thread(target=lambda: good_accumulator.start())
        good_accumulator_thread.daemon = True

        bad_accumulator_thread = threading.Thread(target=lambda: bad_accumulator.start())
        bad_accumulator_thread.daemon = True

        provider_thread.start()
        selector_thread.start()
        good_accumulator_thread.start()
        bad_accumulator_thread.start()

        sleep(1)

        acc_good_imgs = good_accumulator.get_accumulated_protos()
        acc_bad_imgs = bad_accumulator.get_accumulated_protos()

        assert (nr_good_protos == len(acc_good_imgs))
        assert (nr_bad_protos == len(acc_bad_imgs))

        provider.delete_queue(input_queue)
        provider.delete_queue(high_quality_queue_name)
        provider.delete_queue(low_quality_queue_name)

    def test_selector_1(self):
        self.generic_test_selector(2, 10)

    def test_selector_2(self):
        self.generic_test_selector(5, 4)

    def test_selector_3(self):
        self.generic_test_selector(15, 15)

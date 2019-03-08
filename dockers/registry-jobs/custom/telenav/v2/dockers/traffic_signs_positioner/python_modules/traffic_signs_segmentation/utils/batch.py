from collections import namedtuple
from queue import Queue
from threading import Thread, Lock
from time import sleep

from multiprocessing.pool import ThreadPool

import logging.config
import configuration
import image
import utils



BatchElem = namedtuple('Batch', ['image_path', 'scale_factors', 'image_data'])
ImageTransformation = namedtuple(
    'ImageTransformation', ['width', 'height', 'width_crop_size', 'height_crop_size'])

__BatchQueue__ = Queue(maxsize=8)
__FinishBatch__ = False
__FinishBatchLock__ = Lock()

SLEEP_TIME=0.1
NUM_THREADS=8

conf = configuration.load_configuration()

class ImageBatchReader:
    """
    Image reader class transforms the images in the shape required by the networks
    """

    def __init__(self, transformer, lock):
        self.transformer = transformer
        self.lock = lock

    def __call__(self, image_batch):

        batch = []
        logger = logging.getLogger(__name__)
        for image_path in image_batch:
            self.lock.acquire()
            raw_image = image.load_image(image_path)
            logger.info(" Processing image {}.".format(image_path))
            if raw_image is None:
                # Do not skip, better to crash as this should not happen,
                # unless there is a problem with the pipeline
                logger.error("Request path missing {}.".format(image_path))
            cropped_image, width_crop_size, height_crop_size = image.resize_image_fill(raw_image, conf.image_size[1], conf.image_size[0], 3)
            transformed_image = self.transformer.preprocess("data", cropped_image)
            image_transformation = ImageTransformation(raw_image.shape[1], raw_image.shape[0], width_crop_size, height_crop_size)
            batch.append(BatchElem(image_path, image_transformation, transformed_image))
            self.lock.release()
        __BatchQueue__.put(batch)


class BatchReader(Thread):
    """
    Subclass of Thread, specialized in reading batches of images
    """

    def __init__(self, input_path, transformer, batch_size, lock):
        Thread.__init__(self)
        self.input_path = input_path
        self.transformer = transformer
        self.batch_size = batch_size
        self.pool = ThreadPool(NUM_THREADS)
        self.lock = lock

    def run(self):
        images = utils.collect_images(self.input_path)
        images_batches = [images[i:i + self.batch_size]
                          for i in range(0, len(images), self.batch_size)]

        image_reader = ImageBatchReader(self.transformer, self.lock)
        self.pool.map(image_reader, images_batches)
        with __FinishBatchLock__:
            global __FinishBatch__
            __FinishBatch__ = True

        return


class BatchProcessor(Thread):
    """
    Subclass of Thread specialized in processing batches of images
    """

    def __init__(self, batch_handler):
        Thread.__init__(self)
        self.batch_handler = batch_handler

    def __del__(self):
        global __FinishBatch__
        __FinishBatch__ = False

    def __do_work(self):
        batch = __BatchQueue__.get()
        self.batch_handler(batch)
        __BatchQueue__.task_done()

    def run(self):
        while True:
            if not __BatchQueue__.empty():
                self.__do_work()
            else:
                sleep(SLEEP_TIME)
            with __FinishBatchLock__:
                if __FinishBatch__:
                    while not __BatchQueue__.empty():
                        self.__do_work()
                    break

        return

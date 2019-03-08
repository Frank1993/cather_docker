import logging
import threading
import time
from functools import partial
import amqpstorm
from amqpstorm.exception import AMQPError

import apollo_python_common.proto_api as proto_api
import apollo_python_common.io_utils as io_utils
import apollo_python_common.log_util as log_util
import apollo_python_common.sys_util as sys_util

MAX_RETRY_COUNT = 3
MESSAGE_PRIORITY_PROPERTY_KEY = 'priority'


class MessageQueueProvider(object):
    '''
    Generic message queue provider
    '''

    def send_message(self, queue_name, message, properties=None):
        raise NotImplementedError('Method not implemented')

    def start_consuming(self, queue_name, consumer_function):
        raise NotImplementedError('Method not implemented')

    def acknowledge(self, message):
        raise NotImplementedError('Method not implemented')

    def stop_consuming(self, queue_name):
        raise NotImplementedError('Method not implemented')

    def stop_consuming_all(self):
        raise NotImplementedError('Method not implemented')

    def delete_queue(self, queue_name):
        raise NotImplementedError('Method not implemented')


class RabbitMQProvider(MessageQueueProvider):
    '''
    RabbitMQ implementation for MessageQueueProvider
    '''

    QUEUE_PRIORITY_KEY = 'x-max-priority'
    QUEUE_PRIORITY_MAX_VALUE = 5

    def __init__(self, host, port, username, password):
        self._logger = logging.getLogger(__name__)
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._queues_channels = dict()

    @staticmethod
    def __get_new_channel(logger, host, port, username, password):
        logger.info("Connecting to RabbitMQ {}:{}".format(host, port))

        connection = amqpstorm.Connection(host, username, password, port)

        channel = connection.channel()
        logger.info("Connection successful")
        return channel

    @staticmethod
    def __ensure_queue_exists(logger, host, port, username, password, queues_channels, queue_name):
        try:
            if queue_name in queues_channels and (not queues_channels[queue_name].is_open):
                # resetting connection
                queues_channels.pop(queue_name, None)

            if queue_name not in queues_channels:
                channel = RabbitMQProvider.__get_new_channel(logger, host, port, username, password)
                channel.queue.declare(queue_name, durable=True, arguments={RabbitMQProvider.QUEUE_PRIORITY_KEY:
                                                                           RabbitMQProvider.QUEUE_PRIORITY_MAX_VALUE})
                queues_channels[queue_name] = channel
        except Exception as err:
            logger.info("Cannot connect to message queue provider {}:{}".format(host, port))
            logger.error(err)
            sys_util.graceful_shutdown("ensure_queue_exists", logger)

    @staticmethod
    def __start_consuming(logger, host, port, username, password, queues_channels, queue_name, consumer_function,
                          **kwargs):
        try:
            RabbitMQProvider.__ensure_queue_exists(logger, host, port, username, password, queues_channels, queue_name)
            channel = queues_channels[queue_name]
            no_ack = kwargs['no_ack'] if "no_ack" in kwargs else False
            mq_prefetch_count = kwargs['mq_prefetch_count'] if "mq_prefetch_count" in kwargs else 30
            channel.basic.qos(prefetch_count=mq_prefetch_count)
            channel.basic.consume(consumer_function, queue=queue_name, no_ack=no_ack)
            channel.start_consuming()
        except Exception as err:
            logger.info("Cannot connect to message queue provider {}:{}".format(host, port))
            logger.error(err)
            sys_util.graceful_shutdown("start_consuming", logger)

    def send_message(self, queue_name, message, properties=None):
        '''
        Sends a message to the specified queue
        :param queue_name: the queue where to send the message
        :param message: message body
        :return:
        '''
        retry_count = 0
        not_done = True
        while retry_count < MAX_RETRY_COUNT and not_done:
            try:
                RabbitMQProvider.__ensure_queue_exists(self._logger, self._host, self._port, self._username,
                                                       self._password,
                                                       self._queues_channels, queue_name)
                channel = self._queues_channels[queue_name]
                self.__publish_message(channel, message, queue_name, properties)
                not_done = False
            except AMQPError as err:
                self._logger.warning(str(err))
                # resetting connection
                self._queues_channels.pop(queue_name, None)
                retry_count += 1

        if not_done:
            sys_util.graceful_shutdown("send_message", self._logger)

    def __publish_message(self, channel, message_content, queue_name, properties):
        self._logger.debug("Publishing message into queue {}".format(queue_name))

        message = amqpstorm.Message.create(channel, message_content, properties)

        message.publish(queue_name)

    def start_consuming(self, queue_name, consumer_function, **kwargs):
        '''
        Starts consuming messages on the specified queue
        :param queue_name: from where to consume
        :param consumer_function: callback consumer function
        :param kwargs:
        :return:
        '''
        self._logger.info("Start consuming messages on queue {}".format(queue_name))
        target = partial(RabbitMQProvider.__start_consuming, self._logger, self._host, self._port, self._username,
                         self._password,
                         self._queues_channels, queue_name, consumer_function, **kwargs)
        # Starting in a non-blocking way
        mq_receive_thread = threading.Thread(target=target)
        mq_receive_thread.start()

    def acknowledge(self, message):
        '''
        Acknowledge one message consumed
        :param message: message to be ack
        :return:
        '''
        message.ack()

    def stop_consuming(self, queue_name):
        '''
        Stops consuming messages in the specified queue
        :param queue_name: the queue whete to stop
        :return:
        '''
        if queue_name in self._queues_channels:
            channel = self._queues_channels[queue_name]
            channel.stop_consuming()
            channel.close()
            del self._queues_channels[queue_name]
        else:
            self._logger.warning("Cannot stop consuming for queue {}. It don't have any consumer defined.".
                                 format(queue_name))

    def stop_consuming_all(self):
        '''
        Stops consuming messages in all queues
        '''
        for queue_name, channel in list(self._queues_channels.items()):
            self.stop_consuming(queue_name)

    def delete_queue(self, queue_name):
        if queue_name in self._queues_channels:
            channel = self._queues_channels[queue_name]
            channel.queue.delete(queue_name)
            del self._queues_channels[queue_name]


def test_rabbit_mq_provider(conf, consume_it=True):
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    queue_name = conf.mq_input_queue_name
    mq_provider = RabbitMQProvider(conf.mq_host, conf.mq_port, conf.mq_username, conf.mq_password)
    new_message = proto_api.get_new_image_proto("-1", -1,
                                                "/data/traffic_signs_09_2018/test/502165_9d517_33.jpg",
                                                "US",
                                                40.715, -74.011, True)
    logger.info("Sending messages to queue {}".format(queue_name))
    mq_provider.send_message(queue_name, new_message)

    def on_mq_message(message):
        image_proto = proto_api.read_image_proto(message.body)
        logger.info("Receiving messsage with id {} and content {}".format(message.delivery_tag, image_proto))
        mq_provider.acknowledge(message)

    if consume_it:
        mq_provider.start_consuming(queue_name, on_mq_message, no_ack=False)
    idx = 0
    while True:
        time.sleep(0.1)
        mq_provider.send_message(queue_name, new_message)
        logger.info("Publishing new message")
        idx += 1
        if consume_it and idx > 10:
            mq_provider.stop_consuming(queue_name)
            mq_provider.stop_consuming_all()


def test_rabbit_mq_provider_from_folder(conf, folder):
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    queue_name = conf.mq_input_queue_name
    mq_provider = RabbitMQProvider(conf.mq_host, conf.mq_port, conf.mq_username, conf.mq_password)
    for img_file in io_utils.get_images_from_folder(folder):
        new_message = proto_api.get_new_image_proto("-1", -1, img_file, "US",
                                                    40.715, -74.011, True)
        logger.info("Sending messages to queue {}".format(queue_name))
        mq_provider.send_message(queue_name, new_message)


def test_rabbit_mq_provider_for_geotiles(conf, input_file):
    log_util.config(__file__)
    logger = logging.getLogger(__name__)
    queue_name = conf.mq_input_queue_name

    mq_provider = RabbitMQProvider(conf.mq_host, conf.mq_port, conf.mq_username, conf.mq_password)
    geotile_proto = proto_api.get_new_geotile_proto()
    geotile_proto.top_left.latitude = 43.00
    geotile_proto.top_left.longitude = 80.00
    geotile_proto.bottom_right.latitude = 44.00
    geotile_proto.bottom_right.longitude = 81.00
    input_image_set = proto_api.read_imageset_file(input_file)
    for image in input_image_set.images:
        image.metadata.id = "-1"
    geotile_proto.image_set.CopyFrom(input_image_set)

    logger.info("Sending messages to queue {}".format(queue_name))
    mq_provider.send_message(queue_name, geotile_proto.SerializeToString())


if __name__ == '__main__':
    conf = io_utils.config_load('common_mandatory_config.json')
    # conf = io_utils.config_load('/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/config/clustering_config.json')
    test_rabbit_mq_provider(conf, consume_it=False)
    # test_rabbit_mq_provider_from_folder(conf, '/mnt/data/download')
    # test_rabbit_mq_provider_for_geotiles(conf, '/Users/adrianpopovici/Workspace/SL_git/base/python_modules/sign_clustering/data/localization4.bin')

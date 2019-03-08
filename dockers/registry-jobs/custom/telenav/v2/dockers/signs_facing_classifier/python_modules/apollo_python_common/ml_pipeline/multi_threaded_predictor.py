import logging
import queue
import threading
from datetime import datetime
from typing import List
from collections import defaultdict
import traceback
from apollo_python_common.ml_pipeline.message_queue_providers import RabbitMQProvider, MESSAGE_PRIORITY_PROPERTY_KEY
from apollo_python_common.ml_pipeline.message_envelope import MessageEnvelope
import apollo_python_common.proto_api as proto_api
from apollo_python_common.proto_api import MQ_Messsage_Type
import apollo_python_common.audit.ml_pipeline_audit as ml_pipeline_audit
from apollo_python_common.ml_pipeline.config_api import get_config_param, MQ_Param
import apollo_python_common.sys_util as sys_util
from apollo_python_common.audit import init as init_audit


class MultiThreadedPredictor(object):
    '''
    Class able to run the predictions on a proto instance (image/geotile) having separate, synchronized threads for:
    - receive_new_messages
    - pre_process
    - predict
    - postprocess
    - send_new_messages containing predictions
    '''
    ES_KEY_ERROR = 'Error'
    ES_KEY_WARNING = 'Warning'

    def __init__(self, config, mq_message_type: MQ_Messsage_Type = MQ_Messsage_Type.IMAGE, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.mq_message_type: MQ_Messsage_Type = mq_message_type
        self.no_ack = get_config_param(MQ_Param.NO_ACK, config, False)  # Default with ACK
        self.mq_prefetch_count: int = int(get_config_param(MQ_Param.MQ_PREFETCH_COUNT, config, 30))
        self.max_internal_queue_size: int = int(get_config_param(MQ_Param.MAX_INTERNAL_QUEUE_SIZE, config, 30))
        self.mq_provider: RabbitMQProvider = RabbitMQProvider(get_config_param(MQ_Param.MQ_HOST, config),
                                                              get_config_param(MQ_Param.MQ_PORT, config),
                                                              get_config_param(MQ_Param.MQ_USERNAME, config),
                                                              get_config_param(MQ_Param.MQ_PASSWORD, config))
        self.input_queue_name: str = get_config_param(MQ_Param.MQ_INPUT_QUEUE_NAME, config)
        self.input_errors_queue_name: str = get_config_param(MQ_Param.MQ_INPUT_ERRORS_QUEUE_NAME, config)
        self.output_queue_name: str = get_config_param(MQ_Param.MQ_OUTPUT_QUEUE_NAME, config)
        self.predict_batch_size: int = int(get_config_param(MQ_Param.PREDICT_BATCH_SIZE, config, 1))
        self.nr_preprocess_threads: int = int(get_config_param(MQ_Param.MQ_NR_PREPROCESS_THREADS, config, 1))
        self.nr_predict_threads = int(get_config_param(MQ_Param.MQ_NR_PREDICT_THREADS, config, 1))
        self.running = True
        # Queues
        self.q_input_messages: queue.Queue = self.__get_new_queue()
        self.q_preprocessed: queue.Queue = self.__get_new_queue()
        self.q_predicted: queue.Queue = self.__get_new_queue()
        self.q_postprocessed: queue.Queue = self.__get_new_queue()
        # Workers:
        self.thread_preprocess_list = [self.__get_new_thread(self.__preprocess) for _ in
                                       range(self.nr_preprocess_threads)]
        self.thread_predict_list = [self.__get_new_thread(self.__predict) for _ in range(self.nr_predict_threads)]
        self.thread_postprocess = self.__get_new_thread(self.__postprocess)
        self.thread_output = self.__get_new_thread(self.__output)
        init_audit(self.config)
        # {image_name/geotile_key -> dict} containing information which needs to be audited in ES
        self.__audit_dict = defaultdict(dict)

    def set_audit_key_val(self, msg_id, key, val):
        self.__audit_dict[msg_id] = {key: val}

    def __clear_audit(self, msg_id):
        if msg_id in self.__audit_dict:
            del self.__audit_dict[msg_id]

    def __get_new_thread(self, worker_function):
        thread = threading.Thread(target=worker_function)
        thread.daemon = True
        return thread

    def __get_new_queue(self):
        return queue.Queue(maxsize=self.max_internal_queue_size)

    def start(self):
        '''
        Starts predicting based on messages found in the input message queue
        '''
        self.logger.info("Start listening for new input messages")
        self.__start_workers()
        self.__start_receiving_mq_messages()
        for thread in self.thread_preprocess_list:
            thread.join()
        for thread in self.thread_predict_list:
            thread.join()
        self.thread_postprocess.join()
        self.thread_output.join()

    def stop(self):
        self.running = False

    def preprocess(self, msg):
        '''
        Preprocess one proto instance (image/geotile)
        :param msg: message for one raw proto instance
        :return: message for one preprocessed proto instance
        '''
        raise NotImplementedError('Method not implemented')

    def predict(self, msg):
        '''
        Predict one or more proto instance
        :param msg: message for one or more preprocessed proto instances
        :return: one message containing predictions for one or more proto instances
        '''
        raise NotImplementedError('Method not implemented')

    def postprocess(self, msg, proto_instance):
        '''
        Preprocess one proto instance
        :param msg: message containing predictions for one proto instance
        :return: one message containing post-processed predictions for one proto instance
        '''
        raise NotImplementedError('Method not implemented')

    def __start_receiving_mq_messages(self):
        '''
        Receives new messages from input message queue
        '''
        try:
            def on_mq_message(mq_message):
                try:
                    msg_content = self.__get_message_content(mq_message.body)
                    self.logger.debug(
                        "Receiving messsage with id {} and content {}".format(mq_message.delivery_tag, msg_content))
                    msg_env = MessageEnvelope(mq_message, msg_content, msg_content)
                    self.q_input_messages.put(msg_env)
                    self.logger.debug((msg_env.input_message.delivery_tag, "__mq_receive"))
                    ml_pipeline_audit.one_message_was_received(msg_content, self.config)
                except Exception as err:
                    self.logger.exception(err)
                    try:
                        properties = self.__get_message_properties_dict(mq_message)
                        self.mq_provider.send_message(self.input_errors_queue_name, mq_message.body, properties)
                    except Exception as er:
                        self.logger.exception(er)
                        sys_util.graceful_shutdown("on_mq_message", self.logger)

            self.mq_provider.start_consuming(self.input_queue_name, on_mq_message,
                                             no_ack=self.no_ack, mq_prefetch_count=self.mq_prefetch_count)
        except Exception as err:
            self.logger.exception(err)

    def _send_msg_to_error_queue(self, in_msg_env: MessageEnvelope):
        try:
            properties = self.__get_message_properties_dict(in_msg_env.input_message)
            self.mq_provider.send_message(self.input_errors_queue_name, in_msg_env.input_message.body, properties)
            if not self.no_ack:
                self.mq_provider.acknowledge(in_msg_env.input_message)
            tb = traceback.format_exc()
            ml_pipeline_audit.one_message_was_processed(self.__get_message_content(in_msg_env.input_message.body),
                                                        list(), self.config,
                                                        {self.ES_KEY_ERROR: tb})
        except Exception as err:
            self.logger.exception(err)
            sys_util.graceful_shutdown("send_msg_to_error_queue", self.logger)

    def __transform(self, q_input, message_processor_function, q_output, phase_description, call_with_instance_proto):
        while True:
            in_msg_env: MessageEnvelope = q_input.get()
            self.logger.debug((in_msg_env.input_message.delivery_tag, phase_description))
            try:
                start_time = datetime.now().timestamp()
                if call_with_instance_proto:
                    out_msg = message_processor_function(in_msg_env.envelope_content,
                                                         in_msg_env.current_proto)
                else:
                    out_msg = message_processor_function(in_msg_env.envelope_content)
                if phase_description == "preprocess":
                    nr_of_workers = self.nr_preprocess_threads
                else:
                    nr_of_workers = 1
                in_msg_env.processing_time.append((phase_description,
                                                   (round((datetime.now().timestamp() - start_time) / nr_of_workers,
                                                          4))))
                q_output.put(in_msg_env.get_with_new_content(out_msg))
                q_input.task_done()
            except Exception as err:
                self.logger.exception(err)
                q_input.task_done()
                self._send_msg_to_error_queue(in_msg_env)

    def __transform_in_batch(self, q_input, message_processor_function, q_output, phase_description: str):
        batch_accumulator: List[MessageEnvelope] = []
        while True:
            if q_input.qsize() <= 1 or len(batch_accumulator) >= self.predict_batch_size - 1:
                # trigger process:
                in_msg_env: MessageEnvelope = q_input.get()
                should_process_the_batch = True
            else:
                # accumulating:
                in_msg_env = q_input.get()
                should_process_the_batch = False
                q_input.task_done()
            self.logger.debug((in_msg_env.input_message.delivery_tag, phase_description))
            batch_accumulator.append(in_msg_env)
            if should_process_the_batch:
                in_body_msg_list = [in_msg_env.envelope_content for in_msg_env in batch_accumulator]
                try:
                    start_time = datetime.now().timestamp()
                    out_msg_list = message_processor_function(in_body_msg_list)

                    for in_msg_with_env, out_msg in zip(batch_accumulator, out_msg_list):
                        in_msg_with_env.processing_time.append(
                            (phase_description,
                             round((
                                           datetime.now().timestamp() - start_time)
                                   / self.predict_batch_size / self.nr_predict_threads, 4)))
                        q_output.put(in_msg_with_env.get_with_new_content(out_msg))
                    batch_accumulator = list()
                    q_input.task_done()
                except Exception as err:
                    self.logger.exception(err)
                    for in_msg_with_env in batch_accumulator:
                        self._send_msg_to_error_queue(in_msg_with_env)
                    batch_accumulator = list()
                    q_input.task_done()

    def __preprocess(self):
        self.__transform(self.q_input_messages, self.preprocess, self.q_preprocessed, "preprocess", False)

    def __predict(self):
        self.__transform_in_batch(self.q_preprocessed, self.predict, self.q_predicted, "predict")

    def __postprocess(self):
        self.__transform(self.q_predicted, self.postprocess, self.q_postprocessed, "postprocess", True)

    def __output(self):
        '''
        Writes predictions to output message queue
        :return:
        '''
        while True:
            in_msg_env: MessageEnvelope = self.q_postprocessed.get()
            proto_instance = in_msg_env.envelope_content
            proto_instance_serialized = proto_instance.SerializeToString()
            self.logger.debug((in_msg_env.input_message.delivery_tag, "__output"))
            try:
                properties = self.__get_message_properties_dict(in_msg_env.input_message)
                self.mq_provider.send_message(self.output_queue_name, proto_instance_serialized, properties)
                self.q_postprocessed.task_done()
                if not self.no_ack:
                    self.mq_provider.acknowledge(in_msg_env.input_message)

                self.logger.info('Processing time in sec {}'.format(in_msg_env.processing_time))
                ml_pipeline_audit.one_message_was_processed(proto_instance,
                                                            in_msg_env.processing_time,
                                                            self.config,
                                                            self.__audit_dict[
                                                                self.__get_proto_instance_key(proto_instance)])

            except Exception as err:
                self.logger.exception(err)
                self.q_postprocessed.task_done()
                try:
                    self._send_msg_to_error_queue(in_msg_env)
                except Exception as er:
                    self.logger.exception(er)
                    sys_util.graceful_shutdown("output", self.logger)

            self.__clear_audit(self.__get_proto_instance_key(proto_instance))

    def __get_message_properties_dict(self, mq_message):
        return {MESSAGE_PRIORITY_PROPERTY_KEY: mq_message.priority}

    def __start_workers(self):
        for thread in self.thread_preprocess_list:
            thread.start()
        for thread in self.thread_predict_list:
            thread.start()
        self.thread_postprocess.start()
        self.thread_output.start()

    def __get_message_content(self, serialized_content):
        if self.mq_message_type == MQ_Messsage_Type.IMAGE:
            return proto_api.read_image_proto(serialized_content)
        elif self.mq_message_type == MQ_Messsage_Type.GEO_TILE:
            return proto_api.read_geotile_proto(serialized_content)
        else:
            raise Exception('Message type {} is not handled'.format(self.mq_message_type))

    def __get_proto_instance_key(self, proto_instance):
        if self.mq_message_type == MQ_Messsage_Type.IMAGE:
            return proto_instance.metadata.image_path
        elif self.mq_message_type == MQ_Messsage_Type.GEO_TILE:
            return '{} {} {} {}'.format(
                proto_instance.top_left.latitude, proto_instance.top_left.longitude,
                proto_instance.bottom_right.latitude, proto_instance.bottom_right.longitude)
        else:
            raise Exception('Message type {} is not handled'.format(self.mq_message_type))

    # BEWARE: log methods will overwrite messages for same proto_data and key
    def log_audit_warning(self, proto_data, warning_message):
        self.set_audit_key_val(self.__get_proto_instance_key(proto_data), self.ES_KEY_WARNING, warning_message)

    def log_audit_error(self, proto_data, error_message):
        self.set_audit_key_val(self.__get_proto_instance_key(proto_data), self.ES_KEY_ERROR, error_message)

    def log_audit_message(self, proto_data, key, message):
        self.set_audit_key_val(self.__get_proto_instance_key(proto_data), key, message)


class TestMultiThreadedPredictor(MultiThreadedPredictor):
    def preprocess(self, msg):
        return msg

    def predict(self, msg):
        return msg

    def postprocess(self, msg, proto_instance):
        return msg


def test_multi_threaded_predictor(conf_file):
    import apollo_python_common.io_utils as io_utils
    import apollo_python_common.log_util as log_util
    log_util.config(__file__)
    conf = io_utils.config_load(conf_file)
    predictor = TestMultiThreadedPredictor(conf)
    predictor.start()


if __name__ == '__main__':
    test_multi_threaded_predictor('common_mandatory_config.json')

from apollo_python_common.proto_api import MQ_Messsage_Type
from apollo_python_common.mq.abstract_mq_consumer import AbstractMQConsumer


class AccumulationConsumer(AbstractMQConsumer):

    def __init__(self, config, mq_message_type: MQ_Messsage_Type = MQ_Messsage_Type.IMAGE, **kwargs):
        super().__init__(config, mq_message_type=mq_message_type, **kwargs)
        self.config = config
        self.accumulated_protos = []

    def get_accumulated_protos(self):
        return self.accumulated_protos

    def consume_msg(self, message):
        msg_proto = self.get_message_content(message.body)
        self.accumulated_protos.append(msg_proto)
        
        return msg_proto.SerializeToString()

class MessageEnvelope:
    '''
    Envelope for message queue messages received
    '''
    def __init__(self, input_message, envelope_content, current_proto, **kwargs):
        '''
        :param input_message: input_message as it was received from MQ
        :param envelope_content: current message's content
        :param current_proto: proto instance as it was transformed through processing
        :param kwargs: helper optional arguments
        '''
        self.input_message = input_message
        self.envelope_content = envelope_content
        self.current_proto = current_proto
        self.args = kwargs
        self.processing_time = list()

    def get_with_new_content(self, envelope_content) -> 'MessageEnvelope':
        msg = MessageEnvelope(self.input_message, envelope_content, self.current_proto, **self.args)
        msg.processing_time = self.processing_time
        return msg

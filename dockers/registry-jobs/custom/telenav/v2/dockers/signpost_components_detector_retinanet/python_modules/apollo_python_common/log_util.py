import logging
import logging.config
import os
from apollo_python_common.ml_pipeline.config_api import MQ_Param


def config(log_file_name):
    log_file_name = os.path.abspath(log_file_name)
    full_path_log_file = os.path.abspath(os.path.join(os.path.split(log_file_name)[0].split('python_modules')[0],
                                                      "python_modules", "apollo_python_common", "logs", "logging.ini"))
    logging.config.fileConfig(full_path_log_file,
                              defaults={MQ_Param.LOGSTASH_HOST: 'localhost' if MQ_Param.LOGSTASH_HOST not in os.environ else os.environ[MQ_Param.LOGSTASH_HOST],
                                        MQ_Param.LOGSTASH_PORT: '5000' if MQ_Param.LOGSTASH_PORT not in os.environ else os.environ[MQ_Param.LOGSTASH_PORT]
                                        })

import os
import time
import logging

def Logger(level, log_path=None):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter("%(levelname)s:%(module)s:%(lineno)d:%(message)s")

    assert log_path is not None
    time_tag = time.strftime("%Y-%a-%b-%d-%H-%M-%S", time.localtime())
    file_path = 'logs/{}-{}.log'.format(log_path, time_tag)
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
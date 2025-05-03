import logging
import logging.handlers

def setup_logger(log_level=logging.INFO, log_file='./logs/rag.log'):
    # 创建一个logger
    logger = logging.getLogger('logger')
    logger.setLevel(log_level)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.handlers.TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7)
    file_handler.setLevel(log_level)

    # 创建一个handler，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(filename)s] [func: %(funcName)s] [line: %(lineno)d] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# 初始化logger
logger = setup_logger()

# 使用logger
if __name__ == '__main__':
    logger.info('This is an info message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')

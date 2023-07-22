import logging

class Logger_class():
    def __init__(self,path=None) -> None:
        # 创建Logger对象
        logger = logging.getLogger('Logging')
        logger.setLevel(logging.DEBUG)

        if path:
            # 创建文件处理器
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 将格式化器添加到处理器
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到Logger对象
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        self.logger=logger

    def set_logger(self,info):
        self.logger.info(info)

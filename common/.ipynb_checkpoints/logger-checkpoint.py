"""
date: 2024/2/5
author: Zijian Wang
Describe: 日志记录模块。提供日志配置和获取日志记录器的功能，用于记录程序运行过程中的关键信息和错误。
"""
import logging

class Logger:
    """日志记录器类，用于配置和获取日志记录器。

    Attributes:
        level_relation (dict): 不同日志级别与logging模块级别的映射关系。
        logger (logging.Logger): 日志记录器实例。
        filename (str): 日志文件名。
        level (str): 日志级别。
        format (str): 日志格式。
        __NAME__ (str): 日志记录器名称，默认为'cail'。
    """
    level_relation = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    logger = None
    filename = None
    level = None
    format = None

    __NAME__ = 'cail'

    @classmethod
    def set_config(cls, filename=None, level='info', format='%(asctime)s - %(levelname)s: %(message)s')-> None:
        """设置日志记录器的参数。

        Args:
            filename (str): 日志文件名。
            level (str): 日志级别，默认为'info'。
            format (str): 日志格式，默认为'%(asctime)s - %(levelname)s: %(message)s'。
        """
        Logger.filename = filename
        Logger.level = level
        Logger.format = format

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """获取日志记录器实例。
        Args:
            cls：类自己本身。
            
        Return:
            logging.Logger: 配置好的日志记录器实例。
        """
        if not Logger.logger:
            # 如果日志记录器实例不存在或未被创建，则进行以下配置和初始化操作
            
            format_str = logging.Formatter(Logger.format)
            logger = logging.getLogger(Logger.__NAME__)
            logger.setLevel(Logger.level_relation[Logger.level])
            
            sh = logging.StreamHandler()
            # 创建一个输出到控制台的日志处理器
            sh.setFormatter(format_str)
            # 将格式化对象应用到控制台处理器，规定控制台输出的日志格式
            logger.addHandler(sh)
            # 将控制台处理器添加到日志记录器中，实现将日志输出到控制台
            
            fh = logging.FileHandler(Logger.filename)
            # 创建一个输出到文件的日志处理器
            fh.setFormatter(format_str)
            logger.addHandler(fh)
            
            Logger.logger = logger
            # 将配置好的日志记录器实例保存到类属性中，避免重复创建
        return Logger.logger
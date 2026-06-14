# utils/log.py
import logging
import os
import sys


class ProjectLogger:
    # 日志级别关系映射
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self, log_dir="logs", log_level='info'):
        """
        初始化项目日志系统
        :param log_dir: 日志文件存储目录
        :param log_level: 日志级别
        """
        self.log_dir = log_dir
        self.log_level = log_level

        # 确保日志目录存在
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # 配置根日志器
        self._setup_root_logger()

        # 创建不同模块的日志器
        self.system_logger = self._create_logger("system", "system.log")
        self.detection_logger = self._create_logger("detection", "detection.log")
        self.ocr_logger = self._create_logger("ocr", "ocr.log")
        self.stitch_logger = self._create_logger("stitch", "stitch.log")
        self.database_logger = self._create_logger("database", "database.log")
        self.error_logger = self._create_logger("error", "error.log")

    def _setup_root_logger(self):
        """配置根日志器"""
        # 设置根日志器的基本配置
        logging.basicConfig(
            level=self.level_relations.get(self.log_level, logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(os.path.join(self.log_dir, "all.log"), encoding="utf-8")
            ]
        )

    def _create_logger(self, name, filename):
        """
        创建特定模块的日志器
        :param name: 日志器名称
        :param filename: 日志文件名
        :return: 配置好的日志器
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.level_relations.get(self.log_level, logging.INFO))

        # 避免重复添加handler
        if not logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, filename),
                encoding="utf-8",
                mode="a"
            )
            file_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            )
            logger.addHandler(file_handler)

            # 控制台处理器（只添加到根日志器）
            if name == "system":
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
                logger.addHandler(console_handler)

        return logger

    def get_logger(self, module_name):
        """
        获取指定模块的日志器
        :param module_name: 模块名称
        :return: 对应的日志器
        """
        loggers = {
            'system': self.system_logger,
            'detection': self.detection_logger,
            'ocr': self.ocr_logger,
            'stitch': self.stitch_logger,
            'database': self.database_logger,
            'error': self.error_logger
        }
        return loggers.get(module_name, self.system_logger)


# 全局日志实例
_project_logger_instance = None


def get_project_logger(log_dir="logs", log_level='info'):
    """
    获取项目日志实例（单例模式）
    :param log_dir: 日志目录
    :param log_level: 日志级别
    :return: ProjectLogger实例
    """
    global _project_logger_instance
    if _project_logger_instance is None:
        _project_logger_instance = ProjectLogger(log_dir, log_level)
    return _project_logger_instance


def setup_logger_for_module(module_name, log_dir="logs", log_level='info'):
    """
    为特定模块设置日志器
    :param module_name: 模块名称
    :param log_dir: 日志目录
    :param log_level: 日志级别
    :return: 指定模块的日志器
    """
    project_logger = get_project_logger(log_dir, log_level)
    return project_logger.get_logger(module_name)

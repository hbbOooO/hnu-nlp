from typing import Dict, Any, List

import yaml


class YmlLoader:
    def __init__(self, config: Dict[str / List, Any]) -> None:
        """
        yml加载器初始化
        :param config:Dict/List[str,Any] 配置字典
        """
        self.yml_path = config['yml_path']
        # self.default_yml_path = 'common/configs/default.yml'

    def __call__(self) -> Dict:
        """
        特殊方法，可以像函数一样调用自己 用于加载YAML文件并返回解析后的配置字典
        :return: Dict 解析后的配置字典
        """
        config = self._load_yml(self.yml_path)
        return config

    def _load_yml(self, path: str) -> Dict:
        """
        加载YAML文件并返回解析后的配置字典。
        :param path: str YAML文件路径
        :return: Dict 解析后的配置字典
        """
        # with open(self.default_yml_path, encoding='utf-8') as f:
        #     default_content = f.read()
        # default_config = yaml.load(default_content, Loader=yaml.SafeLoader)
        #     读取指定路径的YAML文件
        with open(path, encoding='utf-8') as f:
            content = f.read()
        #     使用yaml.SafeLoader加载YAML内容，获取解析后的配置字典
        config = yaml.load(content, Loader=yaml.SafeLoader)
        # 获取可能存在的另一个YAML文件路径
        load_yml_path = config.get('load_yml_path', None)
        # 如果存在另一个YAML文件路径，递归调用_load_yml方法加载并合并配置
        if load_yml_path is not None:
            load_yml_config = self._load_yml(load_yml_path)
            self.update_config(load_yml_config, config)
            # self.update_config(default_config, load_yml_config)
            return load_yml_config
        else:
            # 如果没有另一个YAML文件路径，直接返回当前加载的配置字典
            # self.update_config(default_config, config)
            return config

    def update_config(self, load_ymk_config: Dict / List, config: Dict / List):
        """
        更新配置字典。递归地将config字典中的内容更新到load_yml_config字典中。
        :param load_ymk_config: Dict/List 需要更新的配置字典
        :param config:Dict/List 提供更新内容的配置字典
        """
        if isinstance(config, dict):
            # 如果config是字典类型，则递归更新字典的每个键值对
            for k, v in config.items():
                if isinstance(v, dict):
                    # 如果值是字典类型，递归更新
                    self.update_config(load_ymk_config[k], config[k])
                elif isinstance(v, list):
                    # 如果列表中的第一个元素是字典或列表类型，递归更新
                    if isinstance(v[0], dict) or isinstance(v[0], list):
                        self.update_config(load_ymk_config[k], config[k])
                    else:
                        # 否则直接赋值
                        load_ymk_config[k] = v
                else:
                    # 否则直接赋值
                    load_ymk_config[k] = v
        elif isinstance(config, list):
            # 如果config是列表类型，则递归更新列表中的每个元素
            for i in range(len(load_ymk_config)):
                if i < len(config):
                    if isinstance(load_ymk_config[i], dict) or isinstance(load_ymk_config[i], list):
                        # 如果列表元素是字典或列表类型，递归更新
                        self.update_config(load_ymk_config[i], config[i])
                else:
                    # 如果config列表比load_yml_config列表长，添加空字典或空列表，并递归更新
                    if isinstance(load_ymk_config[i], dict):
                        config.append({})
                    elif isinstance(load_ymk_config[i], list):
                        config.append([])
                    self.update_config(load_ymk_config[i], config[i])

import yaml

class YmlLoader:
    def __init__(self, config):
        self.yml_path = config['yml_path']
        # self.default_yml_path = 'common/configs/default.yml'

    def __call__(self):
        config = self._load_yml(self.yml_path)
        return config
    
    def _load_yml(self, path):
        # with open(self.default_yml_path, encoding='utf-8') as f:
        #     default_content = f.read()
        # default_config = yaml.load(default_content, Loader=yaml.SafeLoader)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        config = yaml.load(content, Loader=yaml.SafeLoader)
        load_yml_path = config.get('load_yml_path', None)
        if load_yml_path is not None:
            load_yml_config = self._load_yml(load_yml_path)
            self.update_config(load_yml_config, config)
            # self.update_config(default_config, load_yml_config)
            return load_yml_config
        else:
            # self.update_config(default_config, config)
            return config

    def update_config(self, load_ymk_config, config):
        if isinstance(config, dict):
            for k, v in config.items():
                if isinstance(v, dict):
                    self.update_config(load_ymk_config[k], config[k])
                elif isinstance(v, list):
                    if isinstance(v[0], dict) or isinstance(v[0], list):
                        self.update_config(load_ymk_config[k], config[k])
                    else:
                        load_ymk_config[k] = v
                else:
                    load_ymk_config[k] = v
        elif isinstance(config, list):
            for i in range(len(load_ymk_config)):
                if i < len(config):
                    if isinstance(load_ymk_config[i], dict) or isinstance(load_ymk_config[i], list):
                        self.update_config(load_ymk_config[i], config[i])
                else:
                    if isinstance(load_ymk_config[i], dict):
                        config.append({})
                    elif isinstance(load_ymk_config[i], list):
                        config.append([])
                    self.update_config(load_ymk_config[i], config[i])


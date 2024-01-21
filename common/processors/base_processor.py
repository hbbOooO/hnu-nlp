'''
基础预处理器
'''

class IdProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, item, sample):
        sample['id'] = item['id']
        
class LabelProcessor:
    def __init__(self, config):
        self.config = config

    def __call__(self, item, sample):
        if 'label' in item:
            sample['label'] = item['label']
        elif 'labels' in item:
            sample['labels'] = item['labels']
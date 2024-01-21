from common.meters import Meter

class MidMeter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.cider_score = 0
        else:
            self.cider_score = metric['cider_score']
    
    def __str__(self):
        show_str = 'cider_score: {:4f}' \
            .format(self.cider_score)
        return show_str
    
    def __gt__(self, other):
        return self.cider_score > other.cider_score

class Stage2Meter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.cider_score = 0
            self.blue_score = 0
        else:
            self.cider_score = metric['cider_score']
            self.blue_score = metric['blue_score']
        self.avg_score = (2 * self.cider_score + self.blue_score) / 3
    
    def __str__(self):
        show_str = 'avg_score: {:4f}, cider_score: {:4f}, blue_score: {:4f}' \
            .format(self.avg_score, self.cider_score, self.blue_score)
        return show_str
    
    def __gt__(self, other):
        return self.avg_score > other.avg_score
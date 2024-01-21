    
class Meter:
    def __init__(self, metric=None):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()

    # overload operator '>'
    def __gt__(self, other):
        raise NotImplementedError()

class P1Meter(Meter):
    def __init__(self, metric=None):
        self.p1 = metric['p1']
    
    def __str__(self):
        show_str = 'p1: {:.4f}'.format(self.p1)
        return show_str

    def __gt__(self, other):
        return self.p1 > other.p1

class AccMeter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.acc = 0
        else:
            self.acc = metric['acc']
    
    def __str__(self):
        show_str = 'acc: {:.4f}'.format(self.acc)
        return show_str

    def __gt__(self, other):
        return self.acc > other.acc
    

class MacroF1Meter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.macro_f1 = 0
        else:
            self.macro_f1 = metric['macro_f1']
    
    def __str__(self):
        show_str = 'macro_f1: {:.4f}'.format(self.macro_f1)
        return show_str

    def __gt__(self, other):
        return self.macro_f1 > other.macro_f1


class RougeMeter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.rouge_1 = 0
            self.rouge_2 = 0
            self.rouge_L = 0
        else:
            self.rouge_1 = metric['rouge_1']
            self.rouge_2 = metric['rouge_2']
            self.rouge_L = metric['rouge_L']
        self.total_rouge = 0.2 * self.rouge_1 + 0.3* self.rouge_2 + 0.5 * self.rouge_L

    def __str__(self):
        show_str = 'rouge_1: {:.4f}, rouge_2: {:4f}, rouge_L: {:4f}, rouge_total: {:4f}'.format(self.rouge_1, self.rouge_2, self.rouge_L, self.total_rouge)
        return show_str

    def __gt__(self, other):
        return self.total_rouge > other.total_rouge



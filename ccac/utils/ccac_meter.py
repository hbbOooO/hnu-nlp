from common.meters import Meter


class Track2Meter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.bleu = 0
            self.rouge_1 = 0
            self.rouge_2 = 0
            self.rouge_L = 0
        else:
            self.bleu = metric['bleu']
            self.rouge_1 = metric['rouge_1']
            self.rouge_2 = metric['rouge_2']
            self.rouge_L = metric['rouge_L']
    
    def __str__(self):
        show_str = 'bleu: {:.4f}, rouge_1: {:.4f}, rouge_2: {:.4f}, rouge_L: {:.4f}'.format(self.bleu, self.rouge_1, self.rouge_2, self.rouge_L)
        return show_str

    def __gt__(self, other):
        return self.rouge_L > other.rouge_L
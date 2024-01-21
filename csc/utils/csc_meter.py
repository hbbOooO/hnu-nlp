from common.meters import Meter


class CscMeter(Meter):
    def __init__(self, metric=None):
        if metric is None:
            self.detect_accurancy = 0
            self.correct_accurancy = 0
            self.sentence_fpr = 0
            self.detect_precision = 0
            self.detect_recall = 0
            self.detect_f1 = 0
            self.correct_precision = 0
            self.correct_recall = 0
            self.correct_f1 = 0
        else:
            self.detect_accurancy = metric['detect_accurancy']
            self.correct_accurancy = metric['correct_accurancy']
            self.sentence_fpr = metric['sentence_fpr']
            self.detect_precision = metric['detect_precision']
            self.detect_recall = metric['detect_recall']
            self.detect_f1 = metric['detect_f1']
            self.correct_precision = metric['correct_precision']
            self.correct_recall = metric['correct_recall']
            self.correct_f1 = metric['correct_f1']
        self.avg_f1 = (self.detect_f1 + self.correct_f1) / 2
    
    def __str__(self):
        show_str = 'avg_f1: {:.2f}, detect_f1: {:.2f}, correct_f1: {:.2f}, sentence_fpr: {:.2f}, detect_accurancy: {:.2f}, detect_precision: {:.2f}, detect_recall: {:.2f}, correct_accurancy: {:.2f}, correct_precision: {:.2f}, correct_recall: {:.2f}' \
            .format(self.avg_f1, self.detect_f1, self.correct_f1, self.sentence_fpr, self.detect_accurancy, self.detect_precision, self.detect_recall, self.correct_accurancy, self.correct_precision, self.correct_recall)
        return show_str
    
    def __gt__(self, other):
        return self.avg_f1 > other.avg_f1
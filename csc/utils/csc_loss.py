from common.losses import BaseLoss, FocalLoss

class CscLoss(BaseLoss):
    def __init__(self, config):
        super().__init__()
        self.loss_weight = config['loss_weight']
        self.focal_loss = FocalLoss(**config['focal_loss'])

    def __call__(self, *args, **kwargs):
        word_loss = kwargs['word_loss']
        det_label = kwargs['det_label']
        det_prob = kwargs['det_prob']
        det_loss = self.focal_loss(det_prob, det_label)
        loss = self.loss_weight * det_loss + (1 - self.loss_weight) * word_loss
        return loss
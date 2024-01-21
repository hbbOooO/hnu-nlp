from torch import nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, batch):
        if self.training:
            return self.forward_train_base(batch)
        else:
            return self.forward_test_base(batch)

    def forward_train_base(self, batch):
        self.forward_both(batch)
        return self.forward_train(batch)

    def forward_test_base(self, batch):
        self.forward_both(batch)
        return self.forward_test(batch)

    def forward_both(self, batch):
        raise NotImplementedError('forward_both Function is not Implemented in son class of BaseModel')

    def forward_train(self, batch):
        raise NotImplementedError('forward_train Function is not Implemented in son class of BaseModel')
    
    def forward_test(self, batch):
        # raise NotImplementedError('forward_test Function is not Implemented in son class of BaseModel')
        self.forward_train(batch)
import torch

class FGM:
    def __init__(self, model, embed_name, epsilon=1.):
        super(FGM, self).__init__()
        self.model = model
        self.embed_name = embed_name
        self.back = {}
        self.epsilon = epsilon

    def attack(self):
        for name, params in self.model.named_parameters():
            if params.requires_grad and self.embed_name in name:
                self.back[name] = params.data.clone()
                norm = torch.norm(params.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_att = self.epsilon * (params.grad / norm)
                    params.data.add_(r_att)

    def restore(self):
        for name, params in self.model.named_parameters():
            if params.requires_grad and self.embed_name in name:
                assert name in self.back
                params.data = self.back[name]
        self.back = {}
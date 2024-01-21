import torch

class Adversarial:
    def __init__(self, config):
        self.config = config
        


class FGM:
    def __init__(self, model, embed_name, epsilon=0.3):
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

class PGD():
    def __init__(self, model, epsilon=1., alpha=0.3, emb_name='embedding'):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='embedding', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return param_data + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

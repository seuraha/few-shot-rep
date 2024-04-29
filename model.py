import numpy as np
import torch

def get_bound(configs):
    k = configs.get("k")
    d = configs.get("d")
    n1 = configs.get("n1")
    n2 = configs.get("n2")
    nT = configs.get("nT")
    c = configs.get("c")
    delta = configs.get("delta")
    sigma = configs.get("sigma")

    first = k * d * np.log(c * n1) / (c * n1 * nT)
    second = k * np.log(1/delta) * (1/n2)
    return sigma**2 * (first + second)

class TaskPredict(torch.nn.Module):
    def __init__(self, configs):
        super(TaskPredict, self).__init__()
        self.w = torch.nn.Linear(configs.get("k"), 1, bias=True)

    def forward(self, representation):
        return self.w(representation).squeeze()
    
class MTLDLR(torch.nn.Module):
    def __init__(self, configs):
        super(MTLDLR, self).__init__()
        d, k, T = configs.get("d"), configs.get("k"), configs.get("nT")
        self.B = torch.nn.Parameter(torch.randn(d, k))
        self.W = torch.nn.ModuleList([
            TaskPredict(configs) for _ in range(T)
        ])

    def forward(self, X):
        representation = torch.matmul(X, self.B)
        return torch.cat([w(representation[i]) for i, w in enumerate(self.W)])
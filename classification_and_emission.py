import torch
import torch.nn as nn

class ClassificationNet(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(ClassificationNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        # right now assumes x doesn't needto be reshaped
        # just returns the logits
        return self.network(x)

class EmissionNet(nn.Module):
    def __init__(self, in_dim):
        super(EmissionNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 6),
            nn.Tanh()
        )
        # not sure about this initialization
        self.network[0].weight.data.zero_()
        self.network[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],
                                                     dtype=torch.float))

    def forward(self, x):
        x = self.network(x)
        x = torch.stack([
            torch.clamp(x[:,0], 0.0, 1.0), x[:,1], x[:,2], x[:,3], x[:,4],
            x[:,5]], axis=1
        )
        return x

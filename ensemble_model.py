import torch
import torch.nn as nn
import numpy as np



class EnsembleModel(nn.Module):
    def __init__(self, models, input):
        super(EnsembleModel, self).__init__()
        self.models = models
        self.input = input

        self.modelA, self.modelB, self.modelC, self.modelD, self.modelE, self.modelF, self.modelG, self.modelH, self.modelI, self.modelJ = models

        self.fc1 = nn.Linear(input, 5)

    def forward(self, x):
        out = []
        
        out1, out2, out3, out4, out5, out6, out7, out8, out9, out10 = self.modelA(x), self.modelB(x), self.modelC(x), self.modelD(x), self.modelE(x), self.modelF(x), self.modelG(x), self.modelH(x), self.modelI(x), self.modelJ(x),
        
        out = torch.sum(torch.stack([out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]), dim=0)
        out = torch.div(out, len(self.models))
        out = self.fc1(out)

        return out

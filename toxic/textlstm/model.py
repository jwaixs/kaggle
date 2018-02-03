from torch import nn
import torch

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(100, 100)
        self.hidden_start = (
            torch.autograd.Variable(torch.zeros(1, 1, 100)).cuda(),
            torch.autograd.Variable(torch.zeros(1, 1, 100)).cuda()
        )
        self.classifier = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 7)
        )

    def forward(self, x):
        hidden = self.hidden_start
        out = self.hidden_start[0]
        for inp in x:
            out, hidden = self.lstm(inp.view(1, 1, -1), hidden)
        pred = self.classifier(out.view(out.size(0), -1))
        return pred

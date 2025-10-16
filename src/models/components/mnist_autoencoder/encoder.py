from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        # init parent (nn.Module)
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.enc(x)
        return x
    


from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 input_dim = 28*28, 
                 hidden_dim = 64, 
                 latent_dim = 3):
        # init parent (nn.Module)
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        x = self.enc(x)
        return x
    


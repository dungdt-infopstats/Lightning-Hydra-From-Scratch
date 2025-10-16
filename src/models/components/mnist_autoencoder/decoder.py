from torch import nn

class Decoder(nn.Module):
    def __init__(self, 
                 latent_dim = 3, 
                 output_dim = 28*28, 
                 hidden_dim = 64):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.dec(x)
        return x
    

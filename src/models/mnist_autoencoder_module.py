import torch
import lightning as L
import torch.nn.functional as F

class MNISTAutoencoderModule(L.LightningModule):
    def __init__(self, encoder, decoder):
        # init encoder and decoder first
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx, dataloader_idx = None):
        # input, target = batch
        x, _ = batch
        # change the view of tensor (same with np.reshape but do not create a copy,
        # instead change the view to look at the same data with different shape)
        # reshape the view to batch size (first dim = 0) and flat with (-1)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer
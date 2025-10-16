from models.modules.mnist_autoencoder_module import LitAutoEncoder
from data.components.mnist_autoencoder.dataloader import get_dataloaders
import lightning as L

def main():
    model = LitAutoEncoder()
    train_dataloader = get_dataloaders()
    trainer = L.Trainer()

    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()

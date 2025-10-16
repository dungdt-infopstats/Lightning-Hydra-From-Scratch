import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mnist_autoencoder_module import MNISTAutoencoderModule
from src.data.components.mnist_autoencoder.dataloader import get_dataloaders
import lightning as L
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print("="*50)
    print("Configuration:")
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50)    

    model = hydra.utils.instantiate(cfg.model)
    train_loader = hydra.utils.instantiate(cfg.data)

    trainer = L.Trainer(
        max_epochs = 10,
        accelerator = "auto",  # Tự động chọn GPU nếu có, nếu không sẽ dùng CPU
        devices = 1,  # Sử dụng 1 GPU

    )

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()

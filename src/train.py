import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.mnist_autoencoder_module import MNISTAutoencoderModule
from torch.utils.data import DataLoader
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

    model: L.LightningModule = hydra.utils.instantiate(cfg.model)
    train_loader: DataLoader = hydra.utils.instantiate(cfg.data)
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.get("train"):
        trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()

import hydra as hy
from omegaconf import DictConfig, OmegaConf

import torch
from torchvision import transforms, utils as tv_utils
from lightning.pytorch import callbacks, Trainer, LightningModule

from utils import (
    PipelineCheckpoint,
    _fix_hydra_config_serialization,
    ConfigMixin
)
from dm import ImageDatasets

from models.models import Diffusion

# some global stuff necessary for the program
torch.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()


@hy.main(version_base=None, config_path='./configs')
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)  # resolve all string interpolation

    
    model = Diffusion(cfg.models, cfg.training, cfg.inference)
    datamodule = ImageDatasets(cfg.data)

    trainer = Trainer(
        callbacks=[
            callbacks.LearningRateMonitor(
                'epoch', log_momentum=True, log_weight_decay=True),
            PipelineCheckpoint(mode='min', monitor='FID'),
            callbacks.RichProgressBar()
        ],
        logger=hy.utils.instantiate(cfg.logger, _recursive_=True),
        **cfg.pl_trainer
    )
    trainer.fit(model, datamodule=datamodule,
                ckpt_path=cfg.resume_from_checkpoint
                )


if __name__ == '__main__':
    main()
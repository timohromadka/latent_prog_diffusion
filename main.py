import hydra as hy
from omegaconf import DictConfig, OmegaConf


import torch
from torchvision import transforms, utils as tv_utils
from lightning.pytorch import callbacks, Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger


from utils import (
    PipelineCheckpoint,
    _fix_hydra_config_serialization,
    ConfigMixin
)
from dm import ImageDatasets

from models.models import get_model
from args.args import parser, apply_subset_arguments

# some global stuff necessary for the program
torch.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()


def main():
    args = parser.parse_args()
    args = apply_subset_arguments(args)

    model = get_model(args)
    datamodule = ImageDatasets(args)

    # Configure the WandB logger
    wandb_logger = WandbLogger(
        name=args.wandb_run_name,
        project=args.logger_project,
        entity=args.logger_entity,
        save_dir=args.logger_save_dir,
        offline=args.logger_offline,
    )

    model_checkpoint_callback = callbacks.ModelCheckpoint(
        monitor=args.monitor,  # Metric to monitor
        mode=args.mode,  # 'min' if the metric should decrease, 'max' if it should increase
        save_top_k=3,  # Number of top models to save
        dirpath=args.checkpoint_dir,  # Directory to save the models
        filename='{epoch}-{step}-{val_loss:.2f}',  # Filename pattern
    )

    early_stopping_callback = callbacks.EarlyStopping(
        monitor=args.monitor,
        patience=args.patience, 
        mode=args.mode,
    )
    
    # Set up Trainer
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        strategy=args.strategy,
        logger=wandb_logger,
        callbacks=[
            callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
            PipelineCheckpoint(mode='min', monitor='FID'),  # Assuming FID needs to be monitored separately
            callbacks.RichProgressBar(),
            model_checkpoint_callback, 
            early_stopping_callback,
        ],
        num_sanity_val_steps=args.num_sanity_val_steps,
        benchmark=args.benchmark,
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
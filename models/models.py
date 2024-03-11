import hydra as hy
import math
import os
from itertools import chain

from contextlib import contextmanager, nullcontext
from lightning.pytorch import callbacks, Trainer, LightningModule
from omegaconf import DictConfig, OmegaConf
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage as EMA
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim import Adam
from torchvision import transforms, utils as tv_utils
from torchvision.models import vgg16

from diffusers import DDPMPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from utils import (
    PipelineCheckpoint,
    _fix_hydra_config_serialization,
    ConfigMixin
)

torch.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()

class Diffusion(LightningModule):
    def __init__(self,
                 models_cfg: DictConfig,
                 training_cfg: DictConfig,
                 inference_cfg: DictConfig
                 ):
        super().__init__()

        self.training_cfg = training_cfg
        self.inference_cfg = inference_cfg

        self.model = hy.utils.instantiate(models_cfg.unet)
        self.train_scheduler = \
            hy.utils.instantiate(self.training_cfg.scheduler)
        self.infer_scheduler = \
            hy.utils.instantiate(self.inference_cfg.scheduler)

        self.ema = \
            EMA(self.model.parameters(), decay=self.training_cfg.ema_decay) \
            if self.ema_wanted else None

        self._fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self._fid.persistent(mode=True)
        self._fid.requires_grad_(False)

        self.save_hyperparameters()

    @contextmanager
    def metrics(self):
        self._fid.reset()
        yield self
        self._fid.reset()

    @property
    def FID(self):
        return self._fid.compute()

    @property
    def ema_wanted(self):
        return self.training_cfg.ema_decay != -1

    def _fix_hydra_config_serialization(self) -> None:
        for child in chain(self.children(), vars(self).values()):
            if isinstance(child, ConfigMixin):
                _fix_hydra_config_serialization(child)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            checkpoint['ema'] = self.ema.state_dict()
        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if self.ema_wanted:
            self.ema.load_state_dict(checkpoint['ema'])
        return super().on_load_checkpoint(checkpoint)

    def on_before_zero_grad(self, optimizer) -> None:
        if self.ema_wanted:
            self.ema.update(self.model.parameters())
        return super().on_before_zero_grad(optimizer)

    def to(self, *args, **kwargs):
        if self.training_cfg.ema_decay != -1:
            self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def record_data_for_FID(self, batch, real: bool):
        # batch must be either list of PIL Images, ..
        # .. or, a Tensor of shape (BxCxHxW)
        if isinstance(batch, list):
            batch = torch.stack([to_tensor(pil_image)
                              for pil_image in batch], 0)
        self._fid.update(batch.to(dtype=self.dtype, device=self.device),
                         real=real)

    def record_fake_data_for_FID(self, batch):
        self.record_data_for_FID(batch, False)

    def record_real_data_for_FID(self, batch):
        if self.training and self.current_epoch == 0:
            self.record_data_for_FID(batch, True)

    def training_step(self, batch, batch_idx):
        clean_images = batch['images']

        self.record_real_data_for_FID((clean_images + 1) / 2.)

        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            low=0,
            high=self.train_scheduler.config.num_train_timesteps,
            size=(clean_images.size(0), ), device=self.device
        ).long()
        noisy_images = self.train_scheduler.add_noise(
            clean_images, noise, timesteps)

        # Predict the noise residual
        model_output = self.model(noisy_images, timesteps).sample
        loss = torch.nn.functional.mse_loss(model_output, noise)

        log_key = f'{"train" if self.training else "val"}/simple_loss'
        self.log_dict({log_key: loss},
                      prog_bar=True, sync_dist=True,
                      on_step=self.training,
                      on_epoch=not self.training)

        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    @contextmanager
    def maybe_ema(self):
        ema = self.ema  # The EMACallback() ensures this
        ctx = nullcontext if ema is None else ema.average_parameters
        yield ctx

    def sample(self, **kwargs: dict):
        kwargs.pop('output_type', None)
        kwargs.pop('return_dict', False)

        pipe = self.pipeline()

        with self.maybe_ema():
            images, = pipe(
                **kwargs,
                output_type="pil",
                return_dict=False
            )
        return images
    
    def calculate_and_log_fid(self):
        # Generate images for FID calculation
        pil_images = self.sample(batch_size=self.inference_cfg.fid_batch_size, num_samples=self.inference_cfg.num_fid_samples)
        self.record_fake_data_for_FID(pil_images)
        
        # Here, you should ensure that real images are already recorded or record them as needed.
        
        # Compute FID
        fid_score = self.FID
        
        # Log FID score
        self.log('fid', fid_score, on_step=True, on_epoch=False, prog_bar=True, logger=True)


    def pipeline(self) -> DiffusionPipeline:
        pipe = DDPMPipeline(self.model, self.infer_scheduler).to(
            device=self.device, dtype=self.dtype)  # .to() isn't necessary
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def save_pretrained(self, path: str, push_to_hub: bool = False):
        self._fix_hydra_config_serialization()

        pipe = self.pipeline()
        pipe.save_pretrained(path, safe_serialization=True,
                             push_to_hub=push_to_hub)

    def on_validation_epoch_end(self) -> None:
        batch_size = self.inference_cfg.pipeline_kwargs.get(
            'batch_size', self.training_cfg.batch_size * 2)

        n_per_rank = math.ceil(
            self.inference_cfg.num_samples / self.trainer.world_size)
        n_batches_per_rank = math.ceil(
            n_per_rank / batch_size)

        # TODO: This may end up accummulating a little more than given 'n_samples'
        with self.metrics():
            for _ in range(n_batches_per_rank):
                pil_images = self.sample(
                    **self.inference_cfg.pipeline_kwargs
                )
                self.record_fake_data_for_FID(pil_images)

            self.log('FID', self.FID,
                     prog_bar=True, on_epoch=True, sync_dist=True)

        if self.global_rank == 0:
            images = torch.stack([to_tensor(pil_image)
                              for pil_image in pil_images], 0)
            image_grid = tv_utils.make_grid(images,
                                            nrow=math.ceil(batch_size ** 0.5), padding=1)
            try:
                saving_dir = self.logger.experiment.dir  # for wandb
            except AttributeError:
                saving_dir = self.logger.experiment.log_dir  # for TB

            tv_utils.save_image(image_grid,
                                os.path.join(saving_dir, f'samples_epoch_{self.current_epoch}.png'))

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.training_cfg.learning_rate)
        sched = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.99)
        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': sched, 'interval': 'epoch', 'frequency': 1}
        }
        
        
# =====================
# VAE from GitHub
# https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
# =====================
class VAE(LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo
        
# =====================
# Custom VAE
# =====================

# class Encoder(nn.Module):
#     def __init__(self, input_channels=3, latent_dim=2):  # Change for 2D latent space
#         super().__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(input_channels, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())  # Added BatchNorm
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
#         self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
#         self.fc_mu = nn.Linear(256*8*8, latent_dim)
#         self.fc_log_var = nn.Linear(256*8*8, latent_dim)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = x.view(x.size(0), -1)
#         mu = self.fc_mu(x)
#         log_var = self.fc_log_var(x)
#         return mu, log_var

# class Decoder(nn.Module):
#     def __init__(self, output_channels=3, latent_dim=2):  # Change for 2D latent space
#         super().__init__()
#         self.fc = nn.Linear(latent_dim, 256*8*8)
#         self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
#         self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())
#         self.deconv3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU())
#         self.deconv4 = nn.ConvTranspose2d(32, output_channels, 4, 2, 1)

#     def forward(self, z):
#         z = self.fc(z)
#         z = z.view(-1, 256, 8, 8)
#         z = self.deconv1(z)
#         z = self.deconv2(z)
#         z = self.deconv3(z)
#         z = torch.sigmoid(self.deconv4(z))  # Assuming images are normalized to [0,1]
#         return z

# class VAE(LightningModule):
    def __init__(self, input_channels=3, latent_dim=2, learning_rate=1e-3):  # Change for 2D latent space
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.vgg = vgg16(pretrained=True).features[:11].eval()  # For perceptual loss, uses up to ReLU3_1
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def perceptual_loss(self, recon_x, x):
        vgg_recon = self.vgg(recon_x)
        vgg_x = self.vgg(x)
        return F.mse_loss(vgg_recon, vgg_x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self.forward(x)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        perceptual_loss = self.perceptual_loss(recon_x, x)  # Perceptual loss component
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld_loss + perceptual_loss
        self.log_dict({"train_loss": loss, "recon_loss": recon_loss, "kld_loss": kld_loss, "perceptual_loss": perceptual_loss}, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x, mu, log_var = self.forward(x)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        perceptual_loss = self.perceptual_loss(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kld_loss + perceptual_loss
        self.log_dict({"val_loss": loss, "val_recon_loss": recon_loss, "val_kld_loss": kld_loss, "val_perceptual_loss": perceptual_loss}, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
import argparse
import copy
import logging
import glob
import hydra as hy
import math
import os
from itertools import chain

from contextlib import contextmanager, nullcontext
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from lightning.pytorch import callbacks, Trainer, LightningModule
from omegaconf import DictConfig, OmegaConf
# might need to install from source
# known backwards incompatiblity issue
# use command: pip install git+https://github.com/PytorchLightning/lightning-bolts.git@master --upgrade
# here is github forum discussion: https://github.com/Lightning-Universe/lightning-bolts/issues/962
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage as EMA
from torchmetrics import MetricCollection, MeanSquaredError
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.optim import Adam
from torchvision import transforms, utils as tv_utils
from torchvision.utils import save_image
from torchvision.models import vgg16
from tqdm import tqdm

from diffusers import DDPMPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from .vqvae import VQVAE

from utils import (
    PipelineCheckpoint,
    ConfigMixin
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('models/models.py')

torch.set_float32_matmul_precision('medium')
to_tensor = transforms.ToTensor()

def convert_to_rgb(batch):
    if batch.shape[1] == 1: # i.e. single channel image
        batch = batch.repeat(1, 3, 1, 1)  # repeat the channel 3 times
    return batch

def get_model(args: argparse.Namespace):
    if args.model_to_load is None:
        if args.model in ['diffusion', 'latent_diffusion']:
            return Diffusion(args)
        elif args.model == 'vqvae':
            return VQVAELightning(args)
        elif args.model == 'vae':
            return VAELightning(args)
    else:
        load_model_from_run_name(args)
        
def load_model_from_run_name(args):
    """
    Load a model from a given checkpoint path.
    
    Args:
    - teacher_run_name (str): Run name of the teacher model.
    - args: Arguments needed to initialize the model architecture.
    
    Returns:
    - Loaded model.
    """
    logger.info(f'Loading, configuring, and initializing teacher model from checkpoint using run name: {args.run_name_to_load}')
    
    teacher_model_path = os.path.join(args.checkpoint_dir, args.run_name_to_load)
    
    if not os.path.exists(teacher_model_path):
        raise FileNotFoundError(f"Directory not found at {teacher_model_path}")

    checkpoint_files = glob.glob(os.path.join(teacher_model_path, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt files found in {teacher_model_path}")

    # There should only be one, if not, we only grab the first one for simplicity
    checkpoint_path = checkpoint_files[0]

    # Load the checkpoint to access the configuration
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    config = checkpoint.get('config')
    config_dict = vars(config)

    if not config:
        raise ValueError(f"No config found in checkpoint at {checkpoint_path}")
    
    model_args = copy.deepcopy(args)
    
    # dynamically copy all arguments from config to teacher_args
    # this is needed to ensure we can properly load and use the model for inference correctly
    for key, value in vars(config).items():
        setattr(model_args, key, value)

    if model_args.model in ['diffusion', 'latent_diffusion']:
        model_type = Diffusion
    elif model_args.model == 'vae':
        model_type =  VQVAELightning
    else:
        raise ValueError(f"Unsupported model type: {model_args.model}")

    model = model_type.load_from_checkpoint(checkpoint_path, args=model_args)

    return model, model_args    
  
  
class VQVAELightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # Save initialization arguments to log them with TensorBoard or access them later.
        self.args = args
        # Model setup
        self.model = VQVAE(args)
        self.num_fid_samples = args.num_fid_samples
        self.criterion = torch.nn.MSELoss()
        self.beta = args.beta
        self.lr = args.learning_rate 

        # Metrics
        self.metrics = MetricCollection({
            "mse": MeanSquaredError(),
        })
        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self.fid.persistent(mode=True)
        self.fid.requires_grad_(False)
        

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        imgs = batch['images']
        out = self.model(imgs)
        recon_error = self.criterion(out["x_recon"], imgs)
        commitment_loss = self.beta * out["commitment_loss"]
        dictionary_loss = out.get("dictionary_loss", 0)  # This accounts for models not using EMA.
        
        total_loss = recon_error + commitment_loss + dictionary_loss
        return total_loss, recon_error, commitment_loss, dictionary_loss

    def record_data_for_FID(self, batch, real: bool):
        # batch must be either list of PIL Images, ..
        # .. or, a Tensor of shape (BxCxHxW)
        if isinstance(batch, list):
            batch = torch.stack([to_tensor(pil_image)
                              for pil_image in batch], 0)
        self.fid.update(batch.to(dtype=self.dtype, device=self.device),
                         real=real)

    def training_step(self, batch, batch_idx):
        total_loss, recon_error, commitment_loss, dictionary_loss = self.step(batch)
        
        # Log individual losses
        self.log('train_recon_error', recon_error)
        self.log('train_commitment_loss', commitment_loss)
        self.log('train_dictionary_loss', dictionary_loss)
        self.log('train_total_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_error, commitment_loss, dictionary_loss = self.step(batch)
        
        clean_images = batch['images']
        if self.args.calculate_fid:
            # Record real images during the first epoch and for the first num_fid_validation_steps steps
            if self.current_epoch == 0 and batch_idx < (self.args.num_fid_samples/self.args.batch_size):
                if self.args.dataset == 'mnist':
                    clean_images = convert_to_rgb(clean_images)
                self.record_data_for_FID(clean_images, real=True)
                #self.fid.update(clean_images, real=True)
            
        # Log individual losses
        self.log('val_recon_error', recon_error)
        self.log('val_commitment_loss', commitment_loss)
        self.log('val_dictionary_loss', dictionary_loss)
        self.log('val_total_loss', total_loss)

        return total_loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    def generate_images_from_latent_codes(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            # Sample latent codes
            latent_h = self.args.image_size // (2 ** self.model.num_downsampling_layers)
            latent_w = self.args.image_size // (2 ** self.model.num_downsampling_layers)
            sampled_latent_codes = torch.randint(0, self.model.vq.num_embeddings, (num_samples, latent_h, latent_w), device=self.device)
            
            # Directly use the sampled indices to retrieve embeddings from the codebook
            quantized = F.embedding(sampled_latent_codes, self.model.vq.e_i_ts.permute(1, 0))
            quantized = quantized.permute(0, 3, 1, 2)  # shape must be in format: [batch_size, embedding_dim, H, W]

            # Decode the embeddings to generate images
            generated_images = self.model.decoder(quantized)
            
        self.model.train()
        return (generated_images + 1) / 2  # Normalize images to [0, 1]

    def on_validation_epoch_end(self):
        if self.args.calculate_fid:
            # Generate fake images
            
            total_batches = (self.num_fid_samples + self.args.batch_size - 1) // self.args.batch_size
            #for _ in tqdm(range(total_batches), desc=f'Generarating {self.args.num_fid_samples} "fake" samples for FID calculation.'):
            for _ in range(total_batches):
                fake_images = self.generate_images_from_latent_codes(num_samples=self.args.batch_size)
                if self.args.dataset == 'mnist':
                    fake_images = convert_to_rgb(fake_images)
                #self.fid.update(fake_images, real=False)
                self.record_data_for_FID(fake_images, real=False)
                
            logger.info('Calculating FID score!')
            fid_score = self.fid.compute()
            self.log('fid', fid_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            self.fid.reset()
            logger.info('Done!')
      
      
# ==================
# VAE
# ==================      
class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 4, stride=2, padding=1) # Assuming input size is 32x32
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_var = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.conv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(64, output_channels, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 8, 8)
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x))  # Assuming the input & output are normalized to [0,1]
        return x

class VAELightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Encoder(args.input_channels, args.latent_dim)
        self.decoder = Decoder(args.latent_dim, args.input_channels)
        self.criterion = nn.MSELoss()
        self.metric = MeanSquaredError()
        
        # Metrics
        self.metrics = MetricCollection({
            "mse": MeanSquaredError(),
        })
        self.fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self.fid.persistent(mode=True)
        self.fid.requires_grad_(False)
        
    def generate_samples(self, n_samples):
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():  # No need to track gradients
            # Sample from a standard normal distribution
            z = torch.randn(n_samples, self.args.latent_dim, device=self.device)
            # Pass through the decoder
            samples = self.decoder(z)
        self.train()  # Set the model back to training mode
        return samples

    def forward(self, x):
        mu, log_var = self.encoder(x)
        p = torch.randn_like(mu)  # Standard normal distribution
        z = mu + torch.exp(log_var / 2) * p  # Reparameterization trick
        return self.decoder(z), mu, log_var

    def step(self, batch):
        x = batch['images']
        recon, mu, log_var = self(x)
        recon_loss = self.criterion(recon, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + self.args.beta * kld_loss
        return total_loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch)
        self.log('train_loss', total_loss)
        self.log('train_recon_loss', recon_loss)
        self.log('train_kld_loss', kld_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, kld_loss = self.step(batch)
        
        clean_images = batch['images']
        if self.args.calculate_fid:
            # Record real images during the first epoch and for the first num_fid_validation_steps steps
            if self.current_epoch == 0 and batch_idx < (self.args.num_fid_samples/self.args.batch_size):
                if self.args.dataset == 'mnist':
                    clean_images = convert_to_rgb(clean_images)
                self.fid.update(clean_images, real=True)
            
            
        self.log('val_loss', total_loss)
        self.log('val_recon_loss', recon_loss)
        self.log('val_kld_loss', kld_loss)
        self.metric(recon_loss)
        self.log('val_mse', self.metric, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.learning_rate)
    
    def on_validation_epoch_end(self):
        if self.args.calculate_fid:
            # Generate fake images
            
            total_batches = (self.num_fid_samples + self.args.batch_size - 1) // self.args.batch_size
            #for _ in tqdm(range(total_batches), desc=f'Generarating {self.args.num_fid_samples} "fake" samples for FID calculation.'):
            for _ in range(total_batches):
                fake_images = self.generate_images_from_latent_codes(num_samples=self.args.batch_size)
                if self.args.dataset == 'mnist':
                    fake_images = convert_to_rgb(fake_images)
                self.fid.update(fake_images, real=False)
                
            logger.info('Calculating FID score!')
            fid_score = self.fid.compute()
            self.log('fid', fid_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            self.fid.reset()
            logger.info('Done!')
            
# ==================
# DIFFUSION
# ==================
class Diffusion(LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()

        self.args = args
        self.model = self._create_unet_model(args.model_size)
        self.train_scheduler = self._create_scheduler(args, args.train_scheduler)
        self.infer_scheduler = self._create_scheduler(args, args.infer_scheduler)
        
        self.ema = \
            EMA(self.model.parameters(), decay=self.args.ema_decay) \
            if self.ema_wanted else None

        self._fid = FrechetInceptionDistance(
            normalize=True, reset_real_features=False)
        self._fid.persistent(mode=True)
        self._fid.requires_grad_(False)

        self.save_hyperparameters()

    def _create_scheduler(self, args, scheduler_type):
        scheduler_kwargs = {
            "num_train_timesteps": args.num_train_timesteps,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "beta_schedule": args.beta_schedule,
            "trained_betas": None,
            "variance_type": args.variance_type,
            "clip_sample": args.clip_sample,
            "prediction_type": args.prediction_type,
            "thresholding": args.thresholding,
            "dynamic_thresholding_ratio": args.dynamic_thresholding_ratio,
            "clip_sample_range": args.clip_sample_range,
            "sample_max_value": args.sample_max_value,
            "timestep_spacing": args.timestep_spacing,
            "steps_offset": args.steps_offset,
            "rescale_betas_zero_snr": args.rescale_betas_zero_snr
        }
        
        if scheduler_type == 'ddpm':
            return DDPMScheduler(**scheduler_kwargs)
        elif scheduler_type == 'ddim':
            return DDIMScheduler(**scheduler_kwargs)
        else:
            raise ValueError(f"Unsupported scheduler type: {args.scheduler_type}")

    def _create_unet_model(self, model_size):
        if model_size == 'large':
            model = UNet2DModel(
                sample_size=self.args.image_size,
                in_channels=3,
                out_channels=3,
                center_input_sample=False,
                time_embedding_type="positional",
                freq_shift=0,
                flip_sin_to_cos=True,
                down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
                block_out_channels=(128, 128, 256, 256, 512, 512),
                layers_per_block=2,
                mid_block_scale_factor=1,
                downsample_padding=1,
                downsample_type="conv",
                upsample_type="conv",
                dropout=0.0,
                act_fn="silu",
                attention_head_dim=8,
                norm_num_groups=32,
                norm_eps=1e-05,
                add_attention=True,
                resnet_time_scale_shift="default"
            )
        elif model_size == 'small':
            model = UNet2DModel(
                sample_size=16,
                in_channels=3,
                out_channels=3,
                center_input_sample=False,
                time_embedding_type="positional",
                freq_shift=0,
                flip_sin_to_cos=True,
                down_block_types=("DownBlock2D", "AttnDownBlock2D"),
                up_block_types=("AttnUpBlock2D", "UpBlock2D"),
                block_out_channels=(112, 224),
                layers_per_block=1,
                mid_block_scale_factor=1,
                downsample_padding=1,
                downsample_type="conv",
                upsample_type="conv",
                dropout=0.0,
                act_fn="silu",
                attention_head_dim=8,
                norm_num_groups=16,
                norm_eps=1e-05,
                add_attention=True,
                resnet_time_scale_shift="default"
            )
        else:
            raise ValueError("Unsupported model size. Choose 'small' or 'large'.")

        return model
    
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
        return self.args.ema_decay != -1

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
        if self.args.ema_decay != -1:
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
        pil_images = self.sample(batch_size=self.args.batch_size, num_samples=self.args.num_fid_samples)
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
        pipe = self.pipeline()
        pipe.save_pretrained(path, safe_serialization=True,
                             push_to_hub=push_to_hub)

    def on_validation_epoch_end(self) -> None:
        batch_size = self.args.pipeline_kwargs.get(
            'batch_size', self.args.batch_size * 2)

        n_per_rank = math.ceil(
            self.args.num_samples / self.trainer.world_size)
        n_batches_per_rank = math.ceil(
            n_per_rank / batch_size)

        # TODO: This may end up accummulating a little more than given 'n_samples'
        with self.metrics():
            for _ in range(n_batches_per_rank):
                pil_images = self.sample(
                    **self.args.pipeline_kwargs
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
            self.parameters(), lr=self.args.learning_rate)
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
# Lightweight VQ-VAE
# =====================       

        
# =====================
# Custom VAE
# =====================
# class Encoder(nn.Module):
#     def __init__(self, input_channels, image_size, compression_rate, latent_dim):
#         super().__init__()
#         self.compression_rate = compression_rate
#         self.image_size = image_size // compression_rate  # Calculate the compressed image size
#         self.latent_dim = latent_dim
#         # Assume a simple architecture; adjust based on your specific needs
#         self.features = nn.Sequential(
#             nn.Conv2d(input_channels, 32, 3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2, padding=1),
#             nn.ReLU(),
#         )
#         compressed_size = image_size // (2**self.compression_rate)  # Adjust based on your network's actual compression
        
#         self.fc_mu = nn.Linear(64 * compressed_size * compressed_size, latent_dim)
#         self.fc_log_var = nn.Linear(64 * compressed_size * compressed_size, latent_dim)

#     def forward(self, x_orig):
#         x_feat = self.features(x_orig)
#         x = x_feat.view(x_feat.size(0), -1)
#         mu = self.fc_mu(x)
#         log_var = self.fc_log_var(x)
#         return mu, log_var

# class Decoder(nn.Module):
#     def __init__(self, output_channels, image_size, compression_rate, latent_dim):
#         super().__init__()
#         self.image_size = image_size // compression_rate
#         self.latent_dim = latent_dim
#         # Adjust the architecture to match the encoder's inverted process
#         self.fc = nn.Linear(latent_dim, 64 * (self.image_size // 4) * (self.image_size // 4))
#         self.features = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, output_channels, 3, stride=2, padding=1, output_padding=1),
#             nn.Sigmoid(),  # Assuming output in [0,1]
#         )

#     def forward(self, z):
#         z = self.fc(z)
#         z = z.view(-1, 64, self.image_size // 4, self.image_size // 4)
#         z = self.features(z)
#         return z

# class VAE(LightningModule):
#     def __init__(self, args):
#         super().__init__()
#         self.input_channels = args.input_channels
#         self.image_size = args.image_size
#         self.latent_dim = args.latent_dim
#         self.compression_rate = args.compression_rate
#         self.learning_rate = args.learning_rate
        
#         self.encoder = Encoder(self.input_channels, self.image_size, self.compression_rate, self.latent_dim)
#         self.decoder = Decoder(self.input_channels, self.image_size, self.compression_rate, self.latent_dim)
#         # Initialize the VGG16 model for perceptual loss
#         self.vgg = vgg16(pretrained=True).features[:11].eval()
#         for param in self.vgg.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         mu, log_var = self.encoder(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decoder(z), mu, log_var

#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def perceptual_loss(self, recon_x, x):
#         recon_features = self.vgg(recon_x)
#         original_features = self.vgg(x)
#         return F.mse_loss(recon_features, original_features)

#     def training_step(self, batch, batch_idx):
#         x = batch['images']
#         recon_x, mu, log_var = self.forward(x)
#         recon_loss = F.mse_loss(recon_x, x, reduction="mean")
#         kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         perceptual_loss = self.perceptual_loss(recon_x, x)
#         loss = recon_loss + kld_loss + perceptual_loss
#         self.log("train_kld_loss", kld_loss)
#         self.log("train_perceptual_loss", perceptual_loss)
#         self.log("train_recon_loss", recon_loss)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, _ = batch
#         recon_x, mu, log_var = self.forward(x)
#         recon_loss = F.mse_loss(recon_x, x, reduction="mean")
#         kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         perceptual_loss = self.perceptual_loss(recon_x, x)
#         loss = recon_loss + kld_loss + perceptual_loss
#         self.log("val_kld_loss", kld_loss)
#         self.log("val_perceptual_loss", perceptual_loss)
#         self.log("val_recon_loss", recon_loss)
#         self.log("val_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         return Adam(self.parameters(), lr=self.learning_rate)
    
# =======================
# Stable Diffusion VAE
# =======================
# class AutoencoderKL(pl.LightningModule):
#     def __init__(self,
#                  ddconfig,
#                  lossconfig,
#                  embed_dim,
#                  ckpt_path=None,
#                  ignore_keys=[],
#                  image_key="image",
#                  colorize_nlabels=None,
#                  monitor=None,
#                  ema_decay=None,
#                  learn_logvar=False
#                  ):
#         super().__init__()
#         self.learn_logvar = learn_logvar
#         self.image_key = image_key
#         self.encoder = Encoder(**ddconfig)
#         self.decoder = Decoder(**ddconfig)
#         self.loss = instantiate_from_config(lossconfig)
#         assert ddconfig["double_z"]
#         self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
#         self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
#         self.embed_dim = embed_dim
#         if colorize_nlabels is not None:
#             assert type(colorize_nlabels)==int
#             self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
#         if monitor is not None:
#             self.monitor = monitor

#         self.use_ema = ema_decay is not None
#         if self.use_ema:
#             self.ema_decay = ema_decay
#             assert 0. < ema_decay < 1.
#             self.model_ema = LitEma(self, decay=ema_decay)
#             print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

#     def init_from_ckpt(self, path, ignore_keys=list()):
#         sd = torch.load(path, map_location="cpu")["state_dict"]
#         keys = list(sd.keys())
#         for k in keys:
#             for ik in ignore_keys:
#                 if k.startswith(ik):
#                     print("Deleting key {} from state_dict.".format(k))
#                     del sd[k]
#         self.load_state_dict(sd, strict=False)
#         print(f"Restored from {path}")

#     @contextmanager
#     def ema_scope(self, context=None):
#         if self.use_ema:
#             self.model_ema.store(self.parameters())
#             self.model_ema.copy_to(self)
#             if context is not None:
#                 print(f"{context}: Switched to EMA weights")
#         try:
#             yield None
#         finally:
#             if self.use_ema:
#                 self.model_ema.restore(self.parameters())
#                 if context is not None:
#                     print(f"{context}: Restored training weights")

#     def on_train_batch_end(self, *args, **kwargs):
#         if self.use_ema:
#             self.model_ema(self)

#     def encode(self, x):
#         h = self.encoder(x)
#         moments = self.quant_conv(h)
#         posterior = DiagonalGaussianDistribution(moments)
#         return posterior

#     def decode(self, z):
#         z = self.post_quant_conv(z)
#         dec = self.decoder(z)
#         return dec

#     def forward(self, input, sample_posterior=True):
#         posterior = self.encode(input)
#         if sample_posterior:
#             z = posterior.sample()
#         else:
#             z = posterior.mode()
#         dec = self.decode(z)
#         return dec, posterior

#     def get_input(self, batch, k):
#         x = batch[k]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
#         return x

#     def training_step(self, batch, batch_idx, optimizer_idx):
#         inputs = self.get_input(batch, self.image_key)
#         reconstructions, posterior = self(inputs)

#         if optimizer_idx == 0:
#             # train encoder+decoder+logvar
#             aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train")
#             self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
#             self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
#             return aeloss

#         if optimizer_idx == 1:
#             # train the discriminator
#             discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
#                                                 last_layer=self.get_last_layer(), split="train")

#             self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
#             self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
#             return discloss

#     def validation_step(self, batch, batch_idx):
#         log_dict = self._validation_step(batch, batch_idx)
#         with self.ema_scope():
#             log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
#         return log_dict

#     def _validation_step(self, batch, batch_idx, postfix=""):
#         inputs = self.get_input(batch, self.image_key)
#         reconstructions, posterior = self(inputs)
#         aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
#                                         last_layer=self.get_last_layer(), split="val"+postfix)

#         discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
#                                             last_layer=self.get_last_layer(), split="val"+postfix)

#         self.log(f"val{postfix}/rec_loss", log_dict_ae[f"val{postfix}/rec_loss"])
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict

#     def configure_optimizers(self):
#         lr = self.learning_rate
#         ae_params_list = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(
#             self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
#         if self.learn_logvar:
#             print(f"{self.__class__.__name__}: Learning logvar")
#             ae_params_list.append(self.loss.logvar)
#         opt_ae = torch.optim.Adam(ae_params_list,
#                                   lr=lr, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr, betas=(0.5, 0.9))
#         return [opt_ae, opt_disc], []

#     def get_last_layer(self):
#         return self.decoder.conv_out.weight

#     @torch.no_grad()
#     def log_images(self, batch, only_inputs=False, log_ema=False, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         if not only_inputs:
#             xrec, posterior = self(x)
#             if x.shape[1] > 3:
#                 # colorize with random projection
#                 assert xrec.shape[1] > 3
#                 x = self.to_rgb(x)
#                 xrec = self.to_rgb(xrec)
#             log["samples"] = self.decode(torch.randn_like(posterior.sample()))
#             log["reconstructions"] = xrec
#             if log_ema or self.use_ema:
#                 with self.ema_scope():
#                     xrec_ema, posterior_ema = self(x)
#                     if x.shape[1] > 3:
#                         # colorize with random projection
#                         assert xrec_ema.shape[1] > 3
#                         xrec_ema = self.to_rgb(xrec_ema)
#                     log["samples_ema"] = self.decode(torch.randn_like(posterior_ema.sample()))
#                     log["reconstructions_ema"] = xrec_ema
#         log["inputs"] = x
#         return log

#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.conv2d(x, weight=self.colorize)
#         x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
#         return x

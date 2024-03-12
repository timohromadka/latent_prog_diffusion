# training_args.py
import argparse
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser(description="Training arguments for latent prog diffusion")

# Data arguments
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset and also its data directory', choices=['cifar10', 'mnist', '2d'])
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')

# Training arguments
parser.add_argument('--accelerator', type=str, default='gpu', help='Training accelerator')
parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for training')
parser.add_argument('--benchmark', action='store_true', help='If benchmark mode is enabled')
parser.add_argument('--precision', type=str, default='16-mixed', help='Precision for training')
parser.add_argument('--strategy', type=str, default='ddp', help='Distributed data parallel strategy')
parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='Number of sanity validation steps to run')
parser.add_argument('--max_epochs', type=int, default=20, help='Maximum number of training epochs')
parser.add_argument('--enable_model_summary', action='store_true', help='Enable model summary')
parser.add_argument('--log_every_n_steps', type=int, default=10, help='Log metrics every n steps')
parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Check validation set every n epochs')
parser.add_argument('--devices', type=str, default='-1', help='Devices for training')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate. -1 to disable EMA')
parser.add_argument('--monitor', type=str, default='val_total_loss', help='Specify the metric to be monitored for early stopping and model checkpoint callbacks.')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dir', help='Specify the checkpoint directory.')
parser.add_argument('--patience', type=int, default=1, help='Number of epochs to wait after metric stops improving.')
parser.add_argument('--num_train_samples', type=int, default=None, help='Specify amount of training samples to use in the training set. If None, the entire dataset is used.')
parser.add_argument('--num_val_samples', type=int, default=None, help='Specify amount of validation samples to use in the validation set. If None, the entire dataset is used.')
parser.add_argument('--save_top_k', type=int, default=1, help='Specify how many top omdels to saved, according to --monitor metric.')


# Model arguments
parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'latent_diffusion', 'vae'], help='Select which type of model this training run will be for. This in turns specifies the nature/setup of the experiment.')
parser.add_argument('--model_to_load', type=str, help='Specify the model directory for the model to load from..')
parser.add_argument('--center_input_sample', action='store_true', help='Whether to center the input sample')
parser.add_argument('--time_embedding_type', type=str, default='positional', help='Type of time embedding')
parser.add_argument('--freq_shift', type=int, default=0, help='Frequency shift for model')
parser.add_argument('--flip_sin_to_cos', action='store_true', help='Whether to flip sin to cos')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for the model')
parser.add_argument('--act_fn', type=str, default='silu', help='Activation function for the model')
parser.add_argument('--attention_head_dim', type=int, default=8, help='Dimension of attention heads')
parser.add_argument('--norm_num_groups', type=int, default=32, help='Number of groups for normalization layers')
parser.add_argument('--norm_eps', type=float, default=1e-5, help='Epsilon for normalization layers')
parser.add_argument('--add_attention', action='store_true', help='Whether to add attention blocks')

# Logger arguments
parser.add_argument('--wandb_run_name', type=str, default=f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
parser.add_argument('--logger_project', type=str, default='latent_prog_diffusion', help='Project name for WandbLogger')
parser.add_argument('--logger_entity', type=str, default='timohrom', help='Entity name for WandbLogger')
parser.add_argument('--logger_save_dir', type=str, default='./logs', help='Save directory for logger')
parser.add_argument('--logger_offline', action='store_true', help='Run logger in offline mode')

# Scheduler arguments
parser.add_argument('--train_scheduler', type=str, default="ddpm", choices=['dppm', 'ddim'], help='Scheduler type for training.')
parser.add_argument('--infer_scheduler', type=str, default="ddpm", choices=['dppm', 'ddim'], help='Scheduler type for inference.')
parser.add_argument('--num_train_timesteps', type=int, default=1000, help='Number of training timesteps for the scheduler')
parser.add_argument('--beta_start', type=float, default=0.0001, help='Start value for beta in the scheduler')
parser.add_argument('--beta_end', type=float, default=0.02, help='End value for beta in the scheduler')
parser.add_argument('--beta_schedule', type=str, default='squaredcos_cap_v2', choices=['linear', 'squaredcos_cap_v2'], help='Beta schedule type')
parser.add_argument('--trained_betas', type=str, default='linear', help='Array of trained betas')
parser.add_argument('--variance_type', type=str, default='fixed_large', help='Variance type for the scheduler')
parser.add_argument('--clip_sample', action='store_true', help='Whether to clip samples in the scheduler')
parser.add_argument('--prediction_type', type=str, default='epsilon', help='Prediction type for the scheduler')
parser.add_argument('--thresholding', action='store_true', help='Enable thresholding')
parser.add_argument('--dynamic_thresholding_ratio', type=float, default=0.995, help='Dynamic thresholding ratio')
parser.add_argument('--clip_sample_range', type=float, default=1.0, help='Clip sample range')
parser.add_argument('--sample_max_value', type=float, default=1.0, help='Maximum sample value')
parser.add_argument('--timestep_spacing', type=str, default='trailing', help='Timestep spacing for the scheduler')
parser.add_argument('--steps_offset', type=int, default=0, help='Steps offset for the scheduler')
parser.add_argument('--rescale_betas_zero_snr', action='store_true', help='Rescale betas to zero SNR')

# FID Arguments
parser.add_argument('--num_fid_samples', type=int, default=1024, help='Steps offset for the scheduler')
parser.add_argument('--calculate_fid', action='store_true')


# VAE-specific arguments
parser.add_argument('--latent_dim', type=int, default=2, choices=[1,2], help='Number of latent dimensions, either 1 or 2')
parser.add_argument('--compression_rate', type=str, default=4, help='Specify the rate of compression.')
parser.add_argument('--beta', type=float, default=0.25, help='Specify beta.')
parser.add_argument('--num_hiddens', type=int, default=128, help='Number of hidden units in each layer')
parser.add_argument('--num_downsampling_layers', type=int, default=2, help='Number of downsampling layers in the model')
parser.add_argument('--num_residual_layers', type=int, default=2, help='Number of residual layers in the model')
parser.add_argument('--num_residual_hiddens', type=int, default=32, help='Number of hidden units in each residual layer')
parser.add_argument('--embedding_dim', type=int, default=64, help='Dimensionality of the embedding space')
parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings')
parser.add_argument('--use_ema', action='store_true', help='Use exponential moving averages (true/false)')
parser.add_argument('--decay', type=float, default=0.99, help='Decay rate for exponential moving averages')
parser.add_argument('--epsilon', type=float, default=1e-5, help='Epsilon value for numerical stability in calculations')


def apply_subset_arguments(args):
    if args.dataset == "cifar10":
        args.image_size = 32 
        args.input_channels = 3
        args.convert_to_rgb = True
        args.HF_DATASET_IMAGE_KEY = 'img'
    elif args.dataset == "mnist":
        args.image_size = 28
        args.input_channels = 1
        args.HF_DATASET_IMAGE_KEY = 'image'
        args.convert_to_rgb = False
    elif args.dataset == "2d":
        args.image_size = 2
        args.input_channels = 1
        args.HF_DATASET_IMAGE_KEY = None
        args.convert_to_rgb = False
    else:
        raise ValueError(f'Invalid dataset specified: {args.dataset}')
    
    if 'loss' in args.monitor or args.monitor == 'fid':
        args.mode = 'min'
    else:
        args.mode = 'max'
    return args


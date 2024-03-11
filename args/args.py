# training_args.py
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Training arguments for latent prog diffusion")

    # Data arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset and also its data directory', choices=['cifar10', 'mnist', '2d'])
    parser.add_argument('--HF_DATASET_IMAGE_KEY', type=str, default='img', help='Image key in HF dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--image_resolution', type=int, default=32, help='Image resolution for models')
    
    # Training arguments
    parser.add_argument('--accelerator', type=str, default='gpu', help='Training accelerator')
    parser.add_argument('--num_nodes', type=int, default=1, help='Number of nodes for training')
    parser.add_argument('--benchmark', action='store_true', help='If benchmark mode is enabled')
    parser.add_argument('--precision', type=str, default='16-mixed', help='Precision for training')
    parser.add_argument('--strategy', type=str, default='ddp', help='Distributed data parallel strategy')
    parser.add_argument('--num_sanity_val_steps', type=int, default=0, help='Number of sanity validation steps to run')
    parser.add_argument('--max_epochs', type=int, default=3000, help='Maximum number of training epochs')
    parser.add_argument('--enable_model_summary', action='store_true', help='Enable model summary')
    parser.add_argument('--log_every_n_steps', type=int, default=10, help='Log metrics every n steps')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10, help='Check validation set every n epochs')
    parser.add_argument('--devices', type=str, default='-1', help='Devices for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate. -1 to disable EMA')
    
    # Model arguments
    parser.add_argument('--image_size', type=int, default=32, help='Image size (pixel count) for model input')
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
    parser.add_argument('--logger_project', type=str, default='latent_prog_diffusion', help='Project name for WandbLogger')
    parser.add_argument('--logger_entity', type=str, default='timohrom', help='Entity name for WandbLogger')
    parser.add_argument('--logger_save_dir', type=str, default='./logs', help='Save directory for logger')
    parser.add_argument('--logger_offline', action='store_true', help='Run logger in offline mode')
    
    # Scheduler arguments
    parser.add_argument('--num_train_timesteps', type=int, default=1000, help='Number of training timesteps for the scheduler')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Start value for beta in the scheduler')
    parser.add_argument('--beta_end', type=float, default=0.02, help='End value for beta in the scheduler')
    parser.add_argument('--beta_schedule', type=str, default='linear', help='Beta schedule type')
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


    return parser.parse_args()

def apply_subset_arguments(args):
    if args.dataset == "cifar10":
        args.image_size = 32 
        args.num_channels = 3
    elif args.dataset == "mnist":
        args.image_size = 28
        args.num_channels = 1
    else:
        raise ValueError(f'Invalid dataset specified: {args.dataset}')
    return args
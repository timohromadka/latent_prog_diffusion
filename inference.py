# example usage:
# python inference.py --checkpoint_path path/to/your/model_checkpoint.ckpt --n_samples 10 --sample_folder path/to/output_samples

import argparse
import os
import torch
from torchvision.utils import save_image
from models.models import VQVAELightning

def load_vqvae_model(checkpoint_path):
    """
    Load a trained VQVAE model from a given checkpoint path.
    """
    model = VQVAELightning.load_from_checkpoint(checkpoint_path=checkpoint_path)
    model.eval()  # Set the model to evaluation mode
    return model

def generate_novel_samples(model, n_samples):
    """
    Generate novel samples using a trained VQVAE model.
    """
    generated_images = model.generate_images_from_latent_codes(num_samples=n_samples)
    return generated_images

def save_generated_images(images, sample_folder):
    """
    Save generated images to the specified folder.
    """
    os.makedirs(sample_folder, exist_ok=True)
    for i, image in enumerate(images):
        save_image(image, os.path.join(sample_folder, f'sample_{i+1}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples with a trained VQVAE model.')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of novel samples to generate.')
    parser.add_argument('--sample_folder', type=str, required=True, help='Folder to save the generated samples.')
    
    args = parser.parse_args()
    
    model = load_vqvae_model(args.checkpoint_path)
    generated_images = generate_novel_samples(model, args.n_samples)
    save_generated_images(generated_images, args.sample_folder)

    print(f'Successfully generated {args.n_samples} samples and saved to {args.sample_folder}.')

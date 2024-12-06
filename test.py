import numpy as np
from PIL import Image
from torchvision import transforms as TR
import math
import torch
import os
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from ddim_scheduler import DDIMScheduler

def set_timesteps(scheduler, num_inference_steps):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps
    return timesteps


def inverse_loop(latents, timesteps, pipe):
    pass


def ddim_inversion(image_path, pipe, num_inference_steps=100):
    image = Image.open(validation_image)
    image = image.convert("RGB")
    image = image.resize((resolution, resolution), Image.BICUBIC)
    image = (TR.ToTensor()(image).unsqueeze(0)) * 2 - 1
    image = image.to(device=device)
    latents = pipe.vae.encode(image).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    timesteps = set_timesteps(pipe.scheduler, num_inference_steps)
    prev_latents = pipe.scheduler.inverse()


def main():
    device = 'cuda'
    sd_model_path = '/home/cjt/pretrained_models/stable-diffusion-2-1-base'
    vae = AutoencoderKL.from_pretrained(f'{sd_model_path}/vae').to(device)
    unet = UNet2DConditionModel().from_pretrained(f'{sd_model_path}/unet', low_cpu_mem_usage=False, device_map=None).to(device)
    noise_scheduler = DDIMScheduler.from_pretrained(sd_model_path, subfolder="scheduler", 
                                                    num_train_timesteps=300, prediction_type="epsilon")
    pipeline = StableDiffusionPipeline.from_pretrained('/home/cjt/pretrained_models/stable-diffusion-2-1-base',
                                                       vae=vae,
                                                       unet=unet,
                                                       scheduler=noise_scheduler,
                                                       ).to(device)
    vae.eval()
    unet.eval()
    pipeline.eval()
    i = 11
    validation_image = f'/home/cjt/perco/res/Kodak24/vqgan16_vq2cn0313/real/kodim{i}.png'

if __name__ == "__main__":
    main()

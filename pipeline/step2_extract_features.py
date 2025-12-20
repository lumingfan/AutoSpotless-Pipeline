#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import gc
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
from tqdm import tqdm

# Diffusers imports
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

# ==========================================
# Part 1: Modified UNet & Pipeline Classes
# ==========================================

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        deform = None,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if not torch.is_tensor(timesteps):
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()
                if deform is not None:
                    # Deform logic omitted for brevity as it's not used in standard extraction
                    pass

        if self.conv_norm_out:
            sample = self.conv_norm_out(sample)
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        output = {}
        output['up_ft'] = up_ft
        return output, sample


class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        deform = None,
        noise = None
    ):
        device = self._execution_device
        # VAE encode
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        
        if noise is None:
            noise = torch.randn_like(latents).to(device)
        
        # Add noise
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        # UNet forward
        unet_output, noise_pred = self.unet(
            latents_noisy,
            t,
            up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            deform=deform
        )
        
        # Decode (optional, mainly for visualization)
        latents_clean = self.scheduler.step(noise_pred, t, latents_noisy).pred_original_sample
        latents_clean = 1 / self.vae.config.scaling_factor * latents_clean
        image = self.vae.decode(latents_clean).sample

        return unet_output, image


class SDFeaturizer:
    def __init__(self, sd_id='sd2-community/stable-diffusion-2-1', index=1):
        print(f"Loading SD model: {sd_id}...")
        # ✅ 改为 True，使用安全格式
        # 注意：我还加上了 variant="fp16"，这样下载的文件更小，且这符合我们 usually 的显存优化策略
        try:
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", use_safetensors=True, variant="fp16")
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None, use_safetensors=True, variant="fp16")
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler", use_safetensors=True)
        except Exception as e:
            print(f"Failed to load fp16 safetensors, trying default: {e}")
            # 如果 fp16 没有，尝试默认的 safetensors
            unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", use_safetensors=True)
            onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None, use_safetensors=True)
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler", use_safetensors=True)
        onestep_pipe.scheduler.set_timesteps(50)
        
        # Optimization
        onestep_pipe = onestep_pipe.to("cuda")
        onestep_pipe.enable_attention_slicing()
        
        self.pipe = onestep_pipe
        self.gc_collect()

    def gc_collect(self):
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self,
                img_tensor, # single image, [1,c,h,w]
                prompt,
                deform=None,
                t=261,
                up_ft_index=[1],
                ensemble_size=8, 
                noise=None):

        img_tensor = img_tensor.repeat(ensemble_size, 1, 1, 1).cuda() # ensem, c, h, w

        prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=False)[0] # [1, 77, dim]
        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        
        unet_ft_all, image = self.pipe(
            img_tensor=img_tensor,
            t=t,
            up_ft_indices=up_ft_index,
            prompt_embeds=prompt_embeds,
            deform=deform, 
            noise=noise)
        
        fts = []
        mx_shape = 0,0
        for i in up_ft_index:
            unet_ft = unet_ft_all['up_ft'][i] # ensem, c, h, w
            unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
            mx_shape = max(mx_shape[0],unet_ft.shape[-2]), max(mx_shape[0],unet_ft.shape[-1])
            fts += [unet_ft]
            
        fts_resized = []
        for i in range(len(up_ft_index)):
            fts_resized += F.interpolate(fts[i], size=(mx_shape[0], mx_shape[1]), mode='bilinear')

        unet_ft_all = torch.cat(fts_resized,dim=0)  #n,c,h,w
        return unet_ft_all, image

# ==========================================
# Part 2: Main Execution Logic
# ==========================================

def process_images(data_dir, img_size=800):
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    # Note: The notebook uses "SD", let's match it to be safe
    output_dir = data_dir / "SD" 
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Model
    try:
        dift = SDFeaturizer()
    except Exception as e:
        print(f"Failed to load Stable Diffusion. Check internet connection or GPU memory.\nError: {e}")
        return

    filelist = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    print(f"Starting feature extraction for {len(filelist)} images...")
    print(f"Output directory: {output_dir}")
    print(f"Image resize resolution: {img_size}x{img_size}")

    for filename in tqdm(filelist):
        name = os.path.splitext(filename)[0] + '.npy'
        save_path = output_dir / name
        
        if save_path.exists():
            continue

        try:
            # Image Loading & Preprocessing
            img_path = images_dir / filename
            img = PIL.Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))
            
            img_tensor = (torch.tensor(np.array(img)) / 255.0 - 0.5) * 2
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]

            # Extraction
            fts, _ = dift.forward(
                img_tensor,
                prompt='',
                ensemble_size=4, # Reduced from 8 to save VRAM/Time, matches notebook param
                t=261,
                up_ft_index=[1,]
            )

            # Saving
            ft_np = fts.clone().detach().cpu().numpy()
            with open(save_path, "wb") as fout:
                np.save(fout, ft_np)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    print("Extraction complete. Cleaning up GPU memory...")
    
    # Force Cleanup
    del dift
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to the dataset directory (containing 'images' folder)")
    parser.add_argument("--img_size", type=int, default=800, help="Resize dimension for SD input (default: 800)")
    args = parser.parse_args()
    
    process_images(args.data_dir, args.img_size)
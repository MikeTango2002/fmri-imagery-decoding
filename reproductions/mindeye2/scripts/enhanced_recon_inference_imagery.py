#!/usr/bin/env python
"""enhanced_recon_inference.py ported to NSD-Imagery (Section 4.2): the SDXL refiner stage.

Inputs:  evals/{model_name}/{model_name}_all_recons_{mode}.pt (18, gen_rep, 3, 256, 256) and all_predcaptions_{mode}.pt
Outputs: evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt, blended with the low-level ones (--blend_lowlevel)

Verbatim otherwise (same SDXL/ZavyChromaXL refiner, img2img_timepoint=13, guider.scale=5, num_samples=1). Changes:
--mode and --gen_rep; the loop runs over samples outside and the 18 stimuli inside; the ground truth is never
loaded, which makes a leak structurally impossible. The blend: final_evaluations.py blends recon*.75 + blurryrecons*.25
into every refined number (lines 179-181), while this script saves the raw refiner output and the common harness
applies no blend. Default on: the shared1000 run of subject 1 reproduced the published Table 7 column to 0.5 pp.
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

import utils

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)

parser = argparse.ArgumentParser(description="MindEye2 NSD-Imagery refinement (benchmark B)")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--subj", type=int, required=True, choices=[1, 2, 5, 7])
parser.add_argument("--mode", type=str, required=True, choices=["vision", "imagery"])
parser.add_argument("--gen_rep", type=int, default=10)
parser.add_argument("--blend_lowlevel", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

utils.seed_everything(seed)

os.makedirs("evals", exist_ok=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)

all_recons = torch.load(f"evals/{model_name}/{model_name}_all_recons_{mode}.pt")           # (18, R, 3, 256, 256)
all_predcaptions = torch.load(f"evals/{model_name}/{model_name}_all_predcaptions_{mode}.pt",
                              weights_only=False)                                            # (18, R)

n_stim, n_reps_saved = all_recons.shape[0], all_recons.shape[1]
assert n_reps_saved >= gen_rep, \
    f"all_recons_{mode}.pt has {n_reps_saved} reps, --gen_rep is {gen_rep}"
all_recons = all_recons[:, :gen_rep]
all_predcaptions = all_predcaptions[:, :gen_rep]

print(model_name, mode)
print(all_recons.shape, all_predcaptions.shape)

config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
sampler_config = unclip_params["sampler_config"]
sampler_config['params']['num_steps'] = 38
config = OmegaConf.load("generative_models/configs/inference/sd_xl_base.yaml")
config = OmegaConf.to_container(config, resolve=True)
refiner_params = config["model"]["params"]

network_config = refiner_params["network_config"]
denoiser_config = refiner_params["denoiser_config"]
first_stage_config = refiner_params["first_stage_config"]
conditioner_config = refiner_params["conditioner_config"]
scale_factor = refiner_params["scale_factor"]
disable_first_stage_autocast = refiner_params["disable_first_stage_autocast"]

base_ckpt_path = os.path.abspath('mindeyev2_dataset/zavychromaxl_v30.safetensors')
base_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast,
                       ckpt_path=base_ckpt_path)
base_engine.eval().requires_grad_(False)
base_engine.to(device)

base_text_embedder1 = FrozenCLIPEmbedder(
    layer=conditioner_config['params']['emb_models'][0]['params']['layer'],
    layer_idx=conditioner_config['params']['emb_models'][0]['params']['layer_idx'],
)
base_text_embedder1.to(device)

base_text_embedder2 = FrozenOpenCLIPEmbedder2(
    arch=conditioner_config['params']['emb_models'][1]['params']['arch'],
    version=conditioner_config['params']['emb_models'][1]['params']['version'],
    freeze=conditioner_config['params']['emb_models'][1]['params']['freeze'],
    layer=conditioner_config['params']['emb_models'][1]['params']['layer'],
    always_return_pooled=conditioner_config['params']['emb_models'][1]['params']['always_return_pooled'],
    legacy=conditioner_config['params']['emb_models'][1]['params']['legacy'],
)
base_text_embedder2.to(device)

batch = {"txt": "",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
out = base_engine.conditioner(batch)
crossattn = out["crossattn"].to(device)
vector_suffix = out["vector"][:, -1536:].to(device)
print("crossattn", crossattn.shape)
print("vector_suffix", vector_suffix.shape)

batch_uc = {"txt": "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device),
      "target_size_as_tuple": torch.ones(1, 2).to(device) * 1024}
out = base_engine.conditioner(batch_uc)
crossattn_uc = out["crossattn"].to(device)
vector_uc = out["vector"].to(device)

num_samples = 1  # unchanged from upstream, and this is exactly why all_images is never loaded above
img2img_timepoint = 13
base_engine.sampler.guider.scale = 5
def denoiser(x, sigma, c): return base_engine.denoiser(base_engine.model, x, sigma, c)

all_enhancedrecons = torch.zeros(n_stim, gen_rep, 3, 256, 256)

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), base_engine.ema_scope():
    for r in range(gen_rep):
        for img_idx in tqdm(range(n_stim), desc=f"rep {r}"):
            base_engine.sampler.num_steps = 25

            image = transforms.Resize((768, 768))(all_recons[[img_idx], r]).float().to(device)
            prompt = all_predcaptions[img_idx, r]

            assert image.shape[-1] == 768
            z = base_engine.encode_first_stage(image * 2 - 1).repeat(num_samples, 1, 1, 1)

            openai_clip_text = base_text_embedder1(prompt)
            clip_text_tokenized, clip_text_emb = base_text_embedder2(prompt)
            clip_text_emb = torch.hstack((clip_text_emb, vector_suffix))
            clip_text_tokenized = torch.cat((openai_clip_text, clip_text_tokenized), dim=-1)
            c = {"crossattn": clip_text_tokenized.repeat(num_samples, 1, 1), "vector": clip_text_emb.repeat(num_samples, 1)}
            uc = {"crossattn": crossattn_uc.repeat(num_samples, 1, 1), "vector": vector_uc.repeat(num_samples, 1)}

            noise = torch.randn_like(z)
            sigmas = base_engine.sampler.discretization(base_engine.sampler.num_steps).to(device)
            init_z = (z + noise * append_dims(sigmas[-img2img_timepoint], z.ndim)) / torch.sqrt(1.0 + sigmas[0] ** 2.0)
            sigmas = sigmas[-img2img_timepoint:].repeat(num_samples, 1)

            base_engine.sampler.num_steps = sigmas.shape[-1] - 1
            noised_z, _, _, _, c, uc = base_engine.sampler.prepare_sampling_loop(
                init_z, cond=c, uc=uc, num_steps=base_engine.sampler.num_steps)
            for timestep in range(base_engine.sampler.num_steps):
                noised_z = base_engine.sampler.sampler_step(sigmas[:, timestep], sigmas[:, timestep + 1],
                                                            denoiser, noised_z, cond=c, uc=uc, gamma=0)
            samples_x = base_engine.decode_first_stage(noised_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            # num_samples==1, no ground truth loaded: take the sole sample, no selection.
            sample = samples[0].cpu()[None]
            all_enhancedrecons[img_idx, r] = transforms.Resize((256, 256))(sample)[0].float()

if blend_lowlevel:
    all_blurryrecons = torch.load(f"evals/{model_name}/{model_name}_all_blurryrecons_{mode}.pt")[:, :gen_rep]
    all_enhancedrecons = all_enhancedrecons * .75 + all_blurryrecons * .25
    print("blended with the low-level branch (.75 enhanced + .25 blurry, matching final_evaluations.py:181)")

out_path = f"evals/{model_name}/{model_name}_all_enhancedrecons_{mode}.pt"
print("all_enhancedrecons", all_enhancedrecons.shape)
torch.save(all_enhancedrecons, out_path)
print(f"saved {out_path}")

sys.exit(0)

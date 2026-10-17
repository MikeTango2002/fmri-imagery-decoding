#!/usr/bin/env python
"""recon_inference.py ported to NSD-Imagery (Section 4.2): the unrefined reconstructions.

Inputs:  {imagery_dir}/preprocessed_data/subject{subj}/x_{mode}all.pt (18, n_reps, V), the finetuned checkpoint
Outputs: {model_name}_all_{recons,blurryrecons,predcaptions,clipvoxels,seeds}_{mode}.pt

Verbatim otherwise: the configuration Kneeland et al. evaluated. Changes: (1) argparse takes --mode, --imagery_dir,
--gen_rep (10) and --rep_offset (to resume); --new_test is gone. (2) The voxels come from x_{mode}all.pt and
num_voxels from that tensor. (3) The average over repetitions uses voxel.shape[1], not the three hardcoded `3`s
(8 vision / 16 imagery repetitions). (4) Ten full stochastic passes over the same voxels, one seed per pass as in
MIRAGE's recon_inference_mi.py; the seeds are saved. (5) all_clipvoxels is written once; all_blurryrecons is
replicated on the sample axis, as the common harness expects. No selection among samples: the first is taken.
"""

import os
import sys
import argparse
import random
import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
from generative_models.sgm.models.diffusion import DiffusionEngine
from omegaconf import OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

import utils
from models import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
print("device:", device)

# ------------------------------------------------------------------- argparse ---

parser = argparse.ArgumentParser(description="MindEye2 NSD-Imagery reconstruction (benchmark B)")
parser.add_argument("--model_name", type=str, required=True,
                    help="will load ckpt for model found in ../train_logs/model_name")
parser.add_argument("--data_path", type=str, default=os.getcwd())
parser.add_argument("--cache_dir", type=str, default=os.getcwd())
parser.add_argument("--imagery_dir", type=str, default="imagery_dataset",
                    help="output of prepare_nsd_imagery.py")
parser.add_argument("--subj", type=int, required=True, choices=[1, 2, 5, 7])
parser.add_argument("--mode", type=str, required=True, choices=["vision", "imagery"])
parser.add_argument("--gen_rep", type=int, default=10,
                    help="number of full stochastic samples per stimulus")
parser.add_argument("--rep_offset", type=int, default=0,
                    help="resume an interrupted chain: skip the first N samples, which must "
                         "already be on disk")
parser.add_argument("--blurry_recon", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--n_blocks", type=int, default=4)
parser.add_argument("--hidden_dim", type=int, default=2048)
parser.add_argument("--seed", type=int, default=42,
                    help="base seed; only used to seed the run itself, not the per-sample RNG")
args = parser.parse_args()
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

utils.seed_everything(seed)

os.makedirs("evals", exist_ok=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)

NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

# ---------------------------------------------------------------- imagery data ---

voxels_path = f"{imagery_dir}/preprocessed_data/subject{subj}/x_{mode}all.pt"
voxels = torch.load(voxels_path)   # (18, n_reps, V)
assert voxels.shape[0] == 18, f"{voxels_path}: {voxels.shape[0]} stimuli, expected 18"
assert voxels.shape[-1] == NUM_VOXELS[subj], \
    f"{voxels_path}: V={voxels.shape[-1]}, expected {NUM_VOXELS[subj]} for subj0{subj}"
num_voxels = voxels.shape[-1]
n_stim = voxels.shape[0]
print(f"loaded {voxels_path}: {tuple(voxels.shape)} (num_voxels={num_voxels})")

# ---------------------------------------------------------------------- models ---

clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14", version="laion2b_s39b_b160k",
    output_tokens=True, only_tokens=True,
)
clip_img_embedder.to(device)
clip_seq_dim = 256
clip_emb_dim = 1664

if blurry_recon:
    from diffusers import AutoencoderKL
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)


class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()
    def forward(self, x):
        return x

model = MindEyeModule()


class RidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
                torch.nn.Linear(input_size, out_features) for input_size in input_sizes
            ])
    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out

model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from models import BrainNetwork
model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1,
                          clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim)
utils.count_params(model.ridge)
utils.count_params(model.backbone)
utils.count_params(model)

out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim // 52
timesteps = 100

prior_network = PriorNetwork(
        dim=out_dim, depth=depth, dim_head=dim_head, heads=heads,
        causal=False, num_tokens=clip_seq_dim, learned_query_mode="pos_emb",
    )

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network, image_embed_dim=out_dim, condition_on_text_encodings=False,
    timesteps=timesteps, cond_drop_prob=0.2, image_embed_scale=None,
)
model.to(device)
utils.count_params(model.diffusion_prior)
utils.count_params(model)

tag = 'last'
outdir = os.path.abspath(f'../train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir + f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except Exception:
    import deepspeed
    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")

from transformers import AutoProcessor, AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
clip_text_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
clip_text_model.to(device)
clip_text_model.eval().requires_grad_(False)
clip_text_seq_dim = 257
clip_text_emb_dim = 1024


class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super(CLIPConverter, self).__init__()
        self.linear1 = nn.Linear(clip_seq_dim, clip_text_seq_dim)
        self.linear2 = nn.Linear(clip_emb_dim, clip_text_emb_dim)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.linear1(x)
        x = self.linear2(x.permute(0, 2, 1))
        return x

clip_convert = CLIPConverter()
state_dict = torch.load(f"{cache_dir}/bigG_to_L_epoch8.pth", map_location='cpu')['model_state_dict']
clip_convert.load_state_dict(state_dict, strict=True)
clip_convert.to(device)
del state_dict

config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]

first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                       denoiser_config=denoiser_config,
                       first_stage_config=first_stage_config,
                       conditioner_config=conditioner_config,
                       sampler_config=sampler_config,
                       scale_factor=scale_factor,
                       disable_first_stage_autocast=disable_first_stage_autocast)
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

batch = {"jpg": torch.randn(1, 3, 1, 1).to(device),
      "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
      "crop_coords_top_left": torch.zeros(1, 2).to(device)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device)
print("vector_suffix", vector_suffix.shape)

# --------------------------------------------------------------- reconstruction ---

model.to(device)
model.eval().requires_grad_(False)

out_paths = {k: f"evals/{model_name}/{model_name}_all_{k}_{mode}.pt"
            for k in ("recons", "blurryrecons", "predcaptions", "clipvoxels", "seeds")}

if rep_offset > 0:
    assert os.path.exists(out_paths["recons"]), \
        f"--rep_offset={rep_offset} but {out_paths['recons']} does not exist -- nothing to resume"
    prev_recons = torch.load(out_paths["recons"])
    prev_blurry = torch.load(out_paths["blurryrecons"]) if blurry_recon and os.path.exists(out_paths["blurryrecons"]) else None
    prev_caps = torch.load(out_paths["predcaptions"], weights_only=False)
    prev_seeds = torch.load(out_paths["seeds"])
    all_clipvoxels = torch.load(out_paths["clipvoxels"])
    assert prev_recons.shape[1] == rep_offset, \
        f"{out_paths['recons']} has {prev_recons.shape[1]} reps, --rep_offset expects {rep_offset}"
else:
    prev_recons = prev_blurry = prev_caps = prev_seeds = None
    all_clipvoxels = torch.zeros(n_stim, clip_seq_dim, clip_emb_dim)

new_recons = torch.zeros(n_stim, gen_rep, 3, 256, 256)
new_blurry = torch.zeros(n_stim, gen_rep, 3, 256, 256) if blurry_recon else None
new_caps = np.empty((n_stim, gen_rep), dtype=object)
new_seeds = torch.zeros(gen_rep, dtype=torch.long)

with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
    for r in range(gen_rep):
        seed_r = random.randint(0, 10**7)
        utils.seed_everything(seed_r)
        new_seeds[r] = seed_r
        print(f"[subj0{subj} {mode} rep {rep_offset + r}] seed={seed_r}")

        for stim_idx in tqdm(range(n_stim), desc=f"rep {rep_offset + r}"):
            voxel = voxels[stim_idx][None].to(device)   # (1, n_reps, V)
            n_reps = voxel.shape[1]
            for rep in range(n_reps):
                voxel_ridge = model.ridge(voxel[:, [rep]], 0)
                backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
                if rep == 0:
                    clip_voxels = clip_voxels0
                    backbone = backbone0
                    blurry_image_enc = blurry_image_enc0[0]
                else:
                    clip_voxels += clip_voxels0
                    backbone += backbone0
                    blurry_image_enc += blurry_image_enc0[0]
            clip_voxels /= n_reps
            backbone /= n_reps
            blurry_image_enc /= n_reps

            if r == 0 and rep_offset == 0:
                all_clipvoxels[stim_idx] = clip_voxels.cpu()[0]

            prior_out = model.diffusion_prior.p_sample_loop(
                backbone.shape, text_cond=dict(text_embed=backbone),
                cond_scale=1., timesteps=20)

            pred_caption_emb = clip_convert(prior_out)
            generated_ids = clip_text_model.generate(pixel_values=pred_caption_emb, max_length=20)
            generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            new_caps[stim_idx, r] = generated_caption

            samples = utils.unclip_recon(prior_out, diffusion_engine, vector_suffix, num_samples=1)
            new_recons[stim_idx, r] = transforms.Resize((256, 256))(samples.cpu())[0].float()

            if blurry_recon:
                blurred = (autoenc.decode(blurry_image_enc / 0.18215).sample / 2 + 0.5).clamp(0, 1)
                new_blurry[stim_idx, r] = transforms.Resize((256, 256))(blurred.cpu())[0].float()

# ---------------------------------------------------------------------- saving ---

all_recons = torch.cat([prev_recons, new_recons], dim=1) if prev_recons is not None else new_recons
all_caps = np.concatenate([prev_caps, new_caps], axis=1) if prev_caps is not None else new_caps
all_seeds = torch.cat([prev_seeds, new_seeds]) if prev_seeds is not None else new_seeds
if blurry_recon:
    all_blurry = torch.cat([prev_blurry, new_blurry], dim=1) if prev_blurry is not None else new_blurry

print("all_recons", all_recons.shape)
torch.save(all_recons, out_paths["recons"])
if blurry_recon:
    torch.save(all_blurry, out_paths["blurryrecons"])
torch.save(all_caps, out_paths["predcaptions"])
torch.save(all_clipvoxels, out_paths["clipvoxels"])
torch.save(all_seeds, out_paths["seeds"])
print(f"saved {model_name} {mode} outputs ({all_recons.shape[1]} total reps)")

sys.exit(0)

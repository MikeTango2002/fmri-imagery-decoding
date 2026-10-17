#!/usr/bin/env python
# coding: utf-8

# NSD-Imagery branch of Reconstructions.py: reconstructs the stimuli of one subject, condition and seed.
# Inputs:  the tensors of prepare_nsd_imagery.py, shaped (n_stimuli, n_reps, V), and the MindEye1 checkpoints
# Outputs: recons_subj0N_<condition>_seed<S>...pt (best-of-16 picks) and gt_<condition>.pt, in --out_dir
#
# Four changes with respect to Reconstructions.py; the rest (num_voxels, last.pth + epoch120.pth, the Versatile
# Diffusion pipeline, recons_per_sample=16, img2img_strength=.85, the best-of-16 pick) is verbatim, the open-source
# configuration Kneeland et al. evaluated. (1) argparse takes --condition, --seed, --imagery_dir, --out_dir and
# no --data_path. (2) A TensorDataset replaces the webdataset; `torch.mean(voxel,axis=1)` still averages the
# repetitions. (3) The seed is set after parsing; upstream seeds before argparse, and --seed did nothing.
# (4) Output names carry subject, condition and seed. The voxels get no extra normalization.
#
# Usage (from src/ of MindEye1, venv active; VD_DIR = local Versatile Diffusion snapshot):
#   python Reconstructions_imagery.py --subj=1 --condition=imagerysimple --seed=42 \
#       --model_name=prior_257_final_subj01_bimixco_softclip_byol --autoencoder_name=<run name> \
#       --recons_per_sample=16 --imagery_dir=../dataset/preprocessed --out_dir=imagery --vd_cache_dir=$VD_DIR

# In[2]:


import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank = 0
print("device:",device)

import utils
from models import Clipper, OpenClipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

if utils.is_interactive():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# seed is set after argparse (change 3), not here.


# # Configurations

# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--imagery_dir=../dataset/preprocessed \
                    --subj=1 --condition=visioncomplex --seed=42 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol"

    jupyter_args = jupyter_args.split()
    print(jupyter_args)


# In[3]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="None",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--imagery_dir", type=str, default="../dataset/preprocessed",
    help="--out_dir of prepare_nsd_imagery.py (holds subj0N/x_{cond}.pt and gt_{cond}.pt)",
)
parser.add_argument(
    "--out_dir", type=str, default="imagery",
    help="where to write the reconstruction tensors",
)
parser.add_argument(
    "--condition", type=str, default="visioncomplex",
    choices=["visionsimple","visioncomplex","visionconcepts",
             "imagerysimple","imagerycomplex","imageryconcepts"],
    help="modality x stimulus set; visionconcepts is excluded from the benchmark",
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="one of the 10 seeds of the protocol (Kneeland et al., appendix A.2)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=1,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if autoencoder_name=="None":
    autoencoder_name = None

# An unset $VD_DIR expands to the empty string, which argparse accepts: without this
# check the run would try to re-download Versatile Diffusion (~15 GB) into the cwd and
# then die on the scheduler with a confusing offline-mode error.
if not os.path.isdir(vd_cache_dir) or not os.path.exists(
        os.path.join(vd_cache_dir, "model_index.json")):
    sys.exit(f"--vd_cache_dir={vd_cache_dir!r} is not a Versatile Diffusion folder "
             f"(no model_index.json). Did you forget to export VD_DIR?")

# change 3: now that --seed is parsed, seed everything. The rest of the file
# already passes the global `seed` to utils.reconstruction.
utils.seed_everything(seed=seed)
print("condition", condition, "| seed", seed)


# In[4]:


if subj == 1:
    num_voxels = 15724
elif subj == 2:
    num_voxels = 14278
elif subj == 3:
    num_voxels = 15226
elif subj == 4:
    num_voxels = 13153
elif subj == 5:
    num_voxels = 13039
elif subj == 6:
    num_voxels = 17907
elif subj == 7:
    num_voxels = 12682
elif subj == 8:
    num_voxels = 14386
print("subj",subj,"num_voxels",num_voxels)


# In[5]:


# change 2: no webdataset. The tensors come from prepare_nsd_imagery.py and are
# already masked to nsdgeneral, /300 and z-scored per run.
from torch.utils.data import TensorDataset

x_img = torch.load(f"{imagery_dir}/subj0{subj}/x_{condition}.pt").float()   # (6, n_reps, V)
y_img = torch.load(f"{imagery_dir}/gt_{condition}.pt").float()              # (6, 3, 425, 425)
stim_ids = torch.load(f"{imagery_dir}/subj0{subj}/stim_ids_{condition}.pt")

assert x_img.shape[0] == y_img.shape[0] == len(stim_ids), "x / gt / stim_ids length mismatch"
assert x_img.shape[2] == num_voxels, \
    f"{x_img.shape[2]} voxels in x_{condition}.pt, but subj0{subj} needs {num_voxels}"
print("x_img", tuple(x_img.shape), "| y_img", tuple(y_img.shape), "| stim_ids", stim_ids.tolist())

num_val = len(x_img)
batch_size = val_batch_size = 1   # utils.reconstruction asserts n_samples_save==1

val_data = TensorDataset(x_img, y_img, stim_ids)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ## Load autoencoder

# In[6]:


from models import Voxel2StableDiffusionModel

outdir = f'../train_logs/{autoencoder_name}'
ckpt_path = os.path.join(outdir, f'epoch120.pth')

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)

    voxel2sd.load_state_dict(state_dict,strict=False)
    voxel2sd.eval()
    voxel2sd.to(device)
    print("Loaded low-level model!")
else:
    print("No valid path for low-level model specified; not using img2img!") 
    img2img_strength = 1


# # Load VD pipe

# In[12]:


print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
try:
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device).to(torch.float16)
except:
    print("Downloading Versatile Diffusion to", vd_cache_dir)
    vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir = vd_cache_dir).to(device).to(torch.float16)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance 
text_image_ratio = .0 # .5 means equally weight text and image, 0 means use only image
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler


# ## Load Versatile Diffusion model

# In[8]:


img_variations = False

out_dim = 257 * 768
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

out_dim = 768
depth = 6
dim_head = 64
heads = 12 # heads * dim_head = 12 * 64 = 768
timesteps = 100 #100

prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    )

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
)

outdir = f'../train_logs/{model_name}'
ckpt_path = os.path.join(outdir, f'last.pth')

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior.load_state_dict(state_dict,strict=False)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]
pass


# ## Load Image Variations model

# In[9]:


# img_variations = True

# # CLS model
# out_dim = 768
# clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=False, device=device)
# voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
# voxel2clip = BrainNetwork(**voxel2clip_kwargs)
# voxel2clip.requires_grad_(False)
# voxel2clip.eval()

# diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
#     # kwargs for DiffusionPriorNetwork
#     dict(),
#     # kwargs for DiffusionNetwork
#     dict(
#         condition_on_text_encodings=False,
#         timesteps=1000,
#         voxel2clip=voxel2clip,
#     ),
#     voxel2clip_path=None,
# )

# outdir = f'../train_logs/{model_name}'
# ckpt_path = os.path.join(outdir, f'last.pth')

# print("ckpt_path",ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint['model_state_dict']
# print("EPOCH: ",checkpoint['epoch'])
# diffusion_prior.load_state_dict(state_dict,strict=False)
# diffusion_prior.eval().to(device)
# diffusion_priors = [diffusion_prior]
# pass


# In[10]:


# from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler

# sd_cache_dir = '/fsx/home-paulscotti/.cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc'
# unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="unet").to(device)

# unet.eval() # dont want to train model
# unet.requires_grad_(False) # dont need to calculate gradients

# vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(device)
# vae.eval()
# vae.requires_grad_(False)

# noise_scheduler = UniPCMultistepScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
# num_inference_steps = 20


# # Reconstruct one-at-a-time

# In[13]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = False
plotting = False
saving = True
verbose = False
imsize = 512

if img_variations:
    guidance_scale = 7.5
else:
    guidance_scale = 3.5
    
ind_include = np.arange(num_val)
all_brain_recons = None
    
only_lowlevel = False
if img2img_strength == 1:
    img2img = False
elif img2img_strength == 0:
    img2img = True
    only_lowlevel = True
else:
    img2img = True
    
for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=len(ind_include))):
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half()/0.18215).sample / 2 + 0.5

            if val_i==0:
                plt.imshow(utils.torch_to_Image(blurry_recons))
                plt.show()
        else:
            blurry_recons = None

        if only_lowlevel:
            brain_recons = blurry_recons
        else:
            grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip_cls = None, #diffusion_prior_cls.voxel2clip,
                diffusion_priors = diffusion_priors,
                text_token = None,
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_sample = recons_per_sample,
                guidance_scale = guidance_scale,
                img2img_strength = img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps_prior = 100,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                img_variations = img_variations,
                verbose = verbose,
            )

            if plotting:
                plt.show()
                # grid.savefig(f'evals/{model_name}_{val_i}.png')

            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_images = img
        else:
            all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            all_images = torch.vstack((all_images,img))

    if val_i>=np.max(ind_include):
        break

all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# change 4: names parametrized by subject / condition / seed. The ground truth is
# subject independent and already written once by prepare_nsd_imagery.py, but it is
# copied next to the reconstructions so that out_dir is self-contained.
os.makedirs(out_dir, exist_ok=True)
recon_path = (f'{out_dir}/recons_subj0{subj}_{condition}_seed{seed}'
              f'_img2img{img2img_strength}_{recons_per_sample}samples.pt')
gt_path = f'{out_dir}/gt_{condition}.pt'

if saving:
    torch.save(all_brain_recons, recon_path)
    if not os.path.exists(gt_path):
        torch.save(all_images, gt_path)
print(f'recon_path: {recon_path}')

if not utils.is_interactive():
    sys.exit(0)


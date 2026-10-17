#!/usr/bin/env python
# coding: utf-8

# Top-1 image and brain retrieval on shared1000 (the retrieval columns of Table 4.4).
#
# Inputs:  the shared1000 test shards and a trained checkpoint (--data_path, --subj, --model_name)
# Outputs: forward and backward top-1 retrieval with a 95% CI, on screen
#
# The header of Retrievals.py plus its retrieval block, without the LAION part. Upstream calls sys.exit(0)
# before that block when run from the command line; the notebook was the only way to reach it.
#
# Usage (from src/ of MindEye1, venv active):
#   python Retrievals_topk.py --model_name=prior_257_final_subj01_bimixco_softclip_byol --data_path=cache/ --subj=1

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Retrievals.ipynb --to python"
# call(command,shell=True)


# In[1]:


import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
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
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

if utils.is_interactive():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

seed=42
utils.seed_everything(seed=seed)


# # Configurations

# In[3]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    # Example use
    jupyter_args = "--data_path=/fsx/proj-medarc/fmri/natural-scenes-dataset \
                    --subj=1 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol\
                    --model_name2=prior_1x768_final_subj01_bimixco_softclip_byol"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)


# In[4]:


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str,
    help="name of 257x768 model, used for everything except LAION-5B retrieval",
)
parser.add_argument(
    "--model_name2", type=str,
    help="name of 1x768 model, used for LAION-5B retrieval",
)
parser.add_argument(
    "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
    help="Path to where NSD data is stored (see README)",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,5,7],
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)


# In[5]:


# Upstream hardcoded `subj = 1` here, AFTER argparse had already populated the
# globals, silently overriding --subj: subj02/05/07 loaded 15724 voxels and the
# wrong checkpoint, failing on a load_state_dict size mismatch. Commented out so
# --subj is honored.
# subj = 1 #note: we only trained subjects 1 2 5 7, since they have data across full sessions
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




# ---------------------------------------------------------------------------
# Everything above is the header of Retrievals.py (lines 1-120), verbatim.
# Everything below is its "Image/Brain retrieval" block (lines 305-480), verbatim.
# The LAION-5B section in between is skipped: in Retrievals.py it ends with
#   if not utils.is_interactive(): sys.exit(0)
# so this block never runs from the command line; upstream only reaches it
# through Retrievals.ipynb. This file is that block as a script: it prints the top-1 Image and
# Brain retrieval on the shared1000 test set (the retrieval columns of Table 4.4).
# ---------------------------------------------------------------------------


# # Image/Brain retrieval

# In[7]:


batch_size = 300 # same as used in mind_reader

val_url = f"{data_path}/webdataset_avg_split/test/test_subj0{subj}_" + "{0..1}.tar"
meta_url = f"{data_path}/webdataset_avg_split/metadata_subj0{subj}.json"
num_train = 8559 + 300
num_val = 982

val_batch_size = 300
val_loops = 30
voxels_key = 'nsdgeneral.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=True)\
    .decode("torch")\
    .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
    .to_tuple("voxels", "images", "coco")\
    .batched(val_batch_size, partial=False)\
    .with_epoch(val_loops)
val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# check that your data loader is working
for val_i, (voxel, img_input, coco) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape)
    print("img_input.shape",img_input.shape)
    break


# ### VD

# In[8]:


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
timesteps = 100

prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    ).to(device)

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
).to(device)

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


# ### Img variations 

# In[ ]:


# out_dim = 768
# clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=False, device=device)
# voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
# voxel2clip = BrainNetwork(**voxel2clip_kwargs)
# voxel2clip.requires_grad_(False)
# voxel2clip.eval()

# out_dim = 768
# depth = 6
# dim_head = 64
# heads = 12 # heads * dim_head = 12 * 64 = 768
# timesteps = 1000


# diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
#     # kwargs for DiffusionPriorNetwork
#     dict(),
#     # kwargs for DiffusionNetwork
#     dict(
#         condition_on_text_encodings=False,
#         timesteps=timesteps,
#         voxel2clip=voxel2clip,
#     ),
#     voxel2clip_path=None,
# )

# outdir = f'../train_logs/{model_name2}'
# ckpt_path = os.path.join(outdir, f'last.pth')

# print("ckpt_path",ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint['model_state_dict']
# print("EPOCH: ",checkpoint['epoch'])
# diffusion_prior.load_state_dict(state_dict,strict=False)
# diffusion_prior.eval().to(device)
# diffusion_priors = [diffusion_prior]
# pass


# ## Retrieval on test set

# In[9]:


percent_correct_fwds, percent_correct_bwds = [], []
percent_correct_fwd, percent_correct_bwd = None, None

for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl,total=val_loops)):
    with torch.no_grad():
        voxel = torch.mean(voxel,axis=1).to(device) # average across repetitions
        # voxel = voxel[:,np.random.randint(3)].to(device) # random one of the single-trial samples

        emb = clip_extractor.embed_image(img.to(device)).float() # CLIP-Image
        
        _, emb_ = diffusion_prior.voxel2clip(voxel.float()) # CLIP-Brain
        
        # flatten if necessary
        emb = emb.reshape(len(emb),-1)
        emb_ = emb_.reshape(len(emb_),-1)
        
        # l2norm 
        emb = nn.functional.normalize(emb,dim=-1)
        emb_ = nn.functional.normalize(emb_,dim=-1)
        
        labels = torch.arange(len(emb)).to(device)
        bwd_sim = utils.batchwise_cosine_similarity(emb,emb_)  # clip, brain
        fwd_sim = utils.batchwise_cosine_similarity(emb_,emb)  # brain, clip

        assert len(bwd_sim) == batch_size
        
        percent_correct_fwds = np.append(percent_correct_fwds, utils.topk(fwd_sim, labels,k=1).item())
        percent_correct_bwds = np.append(percent_correct_bwds, utils.topk(bwd_sim, labels,k=1).item())
            
        if val_i==0:
            print("Loop 0:",percent_correct_fwds, percent_correct_bwds)
            
percent_correct_fwd = np.mean(percent_correct_fwds)
fwd_sd = np.std(percent_correct_fwds) / np.sqrt(len(percent_correct_fwds))
fwd_ci = stats.norm.interval(0.95, loc=percent_correct_fwd, scale=fwd_sd)

percent_correct_bwd = np.mean(percent_correct_bwds)
bwd_sd = np.std(percent_correct_bwds) / np.sqrt(len(percent_correct_bwds))
bwd_ci = stats.norm.interval(0.95, loc=percent_correct_bwd, scale=bwd_sd)

print(f"fwd percent_correct: {percent_correct_fwd:.4f} 95% CI: [{fwd_ci[0]:.4f},{fwd_ci[1]:.4f}]")
print(f"bwd percent_correct: {percent_correct_bwd:.4f} 95% CI: [{bwd_ci[0]:.4f},{bwd_ci[1]:.4f}]")

fwd_sim = np.array(fwd_sim.cpu())
bwd_sim = np.array(bwd_sim.cpu())

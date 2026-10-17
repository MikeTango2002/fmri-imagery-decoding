"""Versatile Diffusion stage of the NSD-Imagery reconstructions: the final image from the init image and CLIP.

Inputs:  nsd_clip{text,vision}_pred_<cond>.npy and the init images of vdvae_reconstruct_images_imagery.py
Outputs: results/versatile_diffusion_imagery/subjNN/<cond>_seed<S>/<i>.png

Copy of scripts/versatilediffusion_reconstruct_images.py with three changes: the inputs above,
torch.manual_seed(--seed) instead of the fixed 0, and the output path. ddim_steps=50, eta=0, scale=7.5,
strength=0.75, mixing=0.4 are upstream. n_samples stays 1 on purpose: it only feeds an unused `shape`; the
10 samples come from 10 seeds, with both stages reseeded together.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/versatilediffusion_reconstruct_images_imagery.py \
        --sub 1 --condition visioncomplex --seed 0 -diff_str 0.75 -mix_str 0.4
"""

import sys
sys.path.append('versatile_diffusion')
import os
import os.path as osp
import PIL
from PIL import Image
from pathlib import Path
import numpy as np
import numpy.random as npr

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model
from lib.model_zoo.ddim_vd import DDIMSampler_VD
from lib.experiments.sd_default import color_adjust, auto_merge_imlist
from torch.utils.data import DataLoader, Dataset

from lib.model_zoo.vd import VD
from lib.cfg_holder import cfg_unique_holder as cfguh
from lib.cfg_helper import get_command_line_args, cfg_initiates, load_cfg_yaml
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-diff_str", "--diff_str",help="Diffusion Strength",default=0.75)
parser.add_argument("-mix_str", "--mix_str",help="Mixing Strength",default=0.4)
parser.add_argument("-condition", "--condition",help="NSD-Imagery condition",default='visioncomplex')
parser.add_argument("-seed", "--seed",help="Sampling seed (0..9)",type=int,default=0)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
strength = float(args.diff_str)
mixing = float(args.mix_str)
condition = args.condition
assert condition in ['visionsimple','visioncomplex','imagerysimple','imagerycomplex','imageryconcepts']
seed = int(args.seed)


def regularize_image(x):
        BICUBIC = PIL.Image.Resampling.BICUBIC
        if isinstance(x, str):
            x = Image.open(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, PIL.Image.Image):
            x = x.resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, np.ndarray):
            x = PIL.Image.fromarray(x).resize([512, 512], resample=BICUBIC)
            x = tvtrans.ToTensor()(x)
        elif isinstance(x, torch.Tensor):
            pass
        else:
            assert False, 'Unknown image type'

        assert (x.shape[1]==512) & (x.shape[2]==512), \
            'Wrong image size'
        return x

cfgm_name = 'vd_noema'
sampler = DDIMSampler_VD
pth = 'versatile_diffusion/pretrained/vd-four-flow-v1-0-fp16-deprecated.pth'
cfgm = model_cfg_bank()(cfgm_name)
net = get_model()(cfgm)
sd = torch.load(pth, map_location='cpu')
net.load_state_dict(sd, strict=False)


# Might require editing the GPU assignments due to Memory issues
net.clip.cuda(0)
net.autokl.cuda(0)

#net.model.cuda(1)
sampler = sampler(net)
#sampler.model.model.cuda(1)
#sampler.model.cuda(1)
batch_size = 1

pred_text = np.load('data/predicted_features/subj{:02d}/imagery/nsd_cliptext_pred_{}.npy'.format(sub,condition))
pred_text = torch.tensor(pred_text).half().cuda(0)  # single-GPU: was cuda(1), all tensors on the only visible device

pred_vision = np.load('data/predicted_features/subj{:02d}/imagery/nsd_clipvision_pred_{}.npy'.format(sub,condition))
pred_vision = torch.tensor(pred_vision).half().cuda(0)  # single-GPU: was cuda(1)

init_dir = 'results/vdvae_imagery/subj{:02d}/{}_seed{}'.format(sub,condition,seed)
out_dir = 'results/versatile_diffusion_imagery/subj{:02d}/{}_seed{}'.format(sub,condition,seed)
os.makedirs(out_dir, exist_ok=True)


n_samples = 1
ddim_steps = 50
ddim_eta = 0
scale = 7.5
xtype = 'image'
ctype = 'prompt'
net.autokl.half()

# Reseeded per (condition, seed) cell: stochastic_encode is the second noise source of
# the 10 samples per stimulus, and the RNG is global and consumed in image order.
torch.manual_seed(seed)
for im_id in range(len(pred_vision)):

    zim = Image.open('{}/{}.png'.format(init_dir,im_id))

    zim = regularize_image(zim)
    zin = zim*2 - 1
    zin = zin.unsqueeze(0).cuda(0).half()

    init_latent = net.autokl_encode(zin)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)
    #strength=0.75
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    device = 'cuda:0'
    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]).to(device))

    dummy = ''
    utx = net.clip_encode_text(dummy)
    utx = utx.cuda(0).half()  # single-GPU: was cuda(1)

    dummy = torch.zeros((1,3,224,224)).cuda(0)
    uim = net.clip_encode_vision(dummy)
    uim = uim.cuda(0).half()  # single-GPU: was cuda(1)

    z_enc = z_enc.cuda(0)  # single-GPU: was cuda(1)

    h, w = 512,512
    shape = [n_samples, 4, h//8, w//8]

    cim = pred_vision[im_id].unsqueeze(0)
    ctx = pred_text[im_id].unsqueeze(0)

    sampler.model.model.diffusion_model.device='cuda:0'  # single-GPU: was 'cuda:1'
    sampler.model.model.diffusion_model.half().cuda(0)  # single-GPU: was cuda(1)
    #mixing = 0.4

    z = sampler.decode_dc(
        x_latent=z_enc,
        first_conditioning=[uim, cim],
        second_conditioning=[utx, ctx],
        t_start=t_enc,
        unconditional_guidance_scale=scale,
        xtype='image',
        first_ctype='vision',
        second_ctype='prompt',
        mixed_ratio=(1-mixing), )

    z = z.cuda(0).half()
    x = net.autokl_decode(z)
    color_adj='None'
    #color_adj_to = cin[0]
    color_adj_flag = (color_adj!='none') and (color_adj!='None') and (color_adj is not None)
    color_adj_simple = (color_adj=='Simple') or color_adj=='simple'
    color_adj_keep_ratio = 0.5

    if color_adj_flag and (ctype=='vision'):
        x_adj = []
        for xi in x:
            color_adj_f = color_adjust(ref_from=(xi+1)/2, ref_to=color_adj_to)
            xi_adj = color_adj_f((xi+1)/2, keep=color_adj_keep_ratio, simple=color_adj_simple)
            x_adj.append(xi_adj)
        x = x_adj
    else:
        x = torch.clamp((x+1.0)/2.0, min=0.0, max=1.0)
        x = [tvtrans.ToPILImage()(xi) for xi in x]


    x[0].save('{}/{}.png'.format(out_dir,im_id))

print('wrote {} images to {}'.format(len(pred_vision),out_dir))

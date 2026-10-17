"""VDVAE stage of the NSD-Imagery reconstructions: decodes the predicted latents into the init images.

Inputs:  data/predicted_features/subjNN/imagery/nsd_vdvae_pred_<cond>_31l_alpha50k.npy
Outputs: results/vdvae_imagery/subjNN/<cond>_seed<S>/<i>.png

Copy of scripts/vdvae_reconstruct_images.py with four changes: ref_latent comes from a dummy batch (only its
shapes are used, and the 4.3 GB test-stimulus file may be gone); input and counts follow the imagery
prediction; an explicit torch.manual_seed(--seed), because upstream never seeds this script and the 10 samples
would not be reproducible; the output path carries condition and seed. The decode is stochastic (DmolNet.sample).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/vdvae_reconstruct_images_imagery.py \
        --sub 1 --condition visioncomplex --seed 0 -bs 6
"""

import sys
sys.path.append('vdvae')
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import (logger,
                   local_mpi_rank,
                   mpi_size,
                   maybe_download,
                   mpi_rank)
from data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from train_helpers import restore_params
from image_utils import *
from model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=6)
parser.add_argument("-condition", "--condition",help="NSD-Imagery condition",default='visioncomplex')
parser.add_argument("-seed", "--seed",help="Sampling seed (0..9)",type=int,default=0)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)
condition=args.condition
assert condition in ['visionsimple','visioncomplex','imagerysimple','imagerycomplex','imageryconcepts']
seed=int(args.seed)

print('Libs imported')

H = {'image_size': 64, 'image_channels': 3,'seed': 0, 'port': 29500, 'save_dir': './saved_models/test', 'data_root': './', 'desc': 'test', 'hparam_sets': 'imagenet64', 'restore_path': 'imagenet64-iter-1600000-model.th', 'restore_ema_path': 'vdvae/model/imagenet64-iter-1600000-model-ema.th', 'restore_log_path': 'imagenet64-iter-1600000-log.jsonl', 'restore_optimizer_path': 'imagenet64-iter-1600000-opt.th', 'dataset': 'imagenet64', 'ema_rate': 0.999, 'enc_blocks': '64x11,64d2,32x20,32d2,16x9,16d2,8x8,8d2,4x7,4d4,1x5', 'dec_blocks': '1x2,4m1,4x3,8m4,8x7,16m8,16x15,32m16,32x31,64m32,64x12', 'zdim': 16, 'width': 512, 'custom_width_str': '', 'bottleneck_multiple': 0.25, 'no_bias_above': 64, 'scale_encblock': False, 'test_eval': True, 'warmup_iters': 100, 'num_mixtures': 10, 'grad_clip': 220.0, 'skip_threshold': 380.0, 'lr': 0.00015, 'lr_prior': 0.00015, 'wd': 0.01, 'wd_prior': 0.0, 'num_epochs': 10000, 'n_batch': 4, 'adam_beta1': 0.9, 'adam_beta2': 0.9, 'temperature': 1.0, 'iters_per_ckpt': 25000, 'iters_per_print': 1000, 'iters_per_save': 10000, 'iters_per_images': 10000, 'epochs_per_eval': 1, 'epochs_per_probe': None, 'epochs_per_eval_save': 1, 'num_images_visualize': 8, 'num_variables_visualize': 6, 'num_temperatures_visualize': 3, 'mpi_size': 1, 'local_rank': 0, 'rank': 0, 'logdir': './saved_models/test/log'}
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
H = dotdict(H)

H, preprocess_fn = set_up_data(H)

print('Models is Loading')
ema_vae = load_vaes(H)


# ref_latent is only used for the per-layer (c,h,w) shapes in latent_transformation():
# a dummy batch is enough and removes the dependency on nsd_test_stim_sub{N}.npy.
dummy = torch.zeros((2, 64, 64, 3))
data_input, _ = preprocess_fn(dummy)
with torch.no_grad():
    activations = ema_vae.encoder.forward(data_input)
    _, ref_latent = ema_vae.decoder.forward(activations, get_latents=True)

pred_path = 'data/predicted_features/subj{:02d}/imagery/nsd_vdvae_pred_{}_31l_alpha50k.npy'.format(sub,condition)
pred_latents = np.load(pred_path)
print('loaded {} {}'.format(pred_path, pred_latents.shape))

# Transfor latents from flattened representation to hierarchical
def latent_transformation(latents, ref):
  layer_dims = np.array([2**4,2**4,2**8,2**8,2**8,2**8,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**10,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**12,2**14])
  transformed_latents = []
  for i in range(31):
    t_lat = latents[:,layer_dims[:i].sum():layer_dims[:i+1].sum()]
    c,h,w=ref[i]['z'].shape[1:]
    transformed_latents.append(t_lat.reshape(len(latents),c,h,w))
  return transformed_latents

num_images = len(pred_latents)
idx = range(num_images)
input_latent = latent_transformation(pred_latents[idx],ref_latent)


def sample_from_hier_latents(latents,sample_ids):
  sample_ids = [id for id in sample_ids if id<len(latents[0])]
  layers_num=len(latents)
  sample_latents = []
  for i in range(layers_num):
    sample_latents.append(torch.tensor(latents[i][sample_ids]).float().cuda())
  return sample_latents

out_dir = 'results/vdvae_imagery/subj{:02d}/{}_seed{}'.format(sub,condition,seed)
os.makedirs(out_dir, exist_ok=True)

# Deliberate deviation from upstream, see the module docstring: the decode is stochastic and
# the 10 samples per stimulus must be reproducible. The RNG is global and consumed in
# sequence; a condition must be reconstructed in one process, in fixed index order.
torch.manual_seed(seed)

for i in range(int(np.ceil(num_images/batch_size))):
  print(i*batch_size)
  samp = sample_from_hier_latents(input_latent,range(i*batch_size,(i+1)*batch_size))
  px_z = ema_vae.decoder.forward_manual_latents(len(samp[0]), samp, t=None)
  sample_from_latent = ema_vae.decoder.out_net.sample(px_z)
  for j in range(len(sample_from_latent)):
      im = sample_from_latent[j]
      im = Image.fromarray(im)
      im = im.resize((512,512),resample=3)
      im.save('{}/{}.png'.format(out_dir,i*batch_size+j))

print('wrote {} images to {}'.format(num_images,out_dir))

#!/usr/bin/env python
# coding: utf-8

"""Batch driver for Reconstructions_imagery.py: the whole (subject, condition, seed) matrix in one process.

Inputs and outputs: those of Reconstructions_imagery.py.

One process instead of 200: the single-shot script reloads Versatile Diffusion (~15 GB), CLIP ViT-L/14, the
prior and the autoencoder on every call, about half of the wall-clock. Here Versatile Diffusion and CLIP load
once, and the prior (last.pth) and autoencoder (epoch120.pth) only when the subject changes. The numbers do not
change: utils.reconstruction() builds a fresh torch.Generator seeded at every call (utils.py:508-509) and every
draw takes it explicitly; the state left by the previous run is irrelevant (one configuration run both ways
matched under torch.equal). The recipe is copied verbatim from Reconstructions_imagery.py.

Usage (from src/ of MindEye1, venv active):
    CUDA_VISIBLE_DEVICES=0 python Reconstructions_imagery_batch.py --subj 1 2 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --vd_cache_dir=$VD_DIR --imagery_dir=../dataset/preprocessed --out_dir=imagery
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

NUM_VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
              5: 13039, 6: 17907, 7: 12682, 8: 14386}

ALL_CONDITIONS = ["visioncomplex", "visionsimple", "imagerycomplex", "imagerysimple", "imageryconcepts"]


parser = argparse.ArgumentParser(description="NSD-Imagery reconstructions, whole matrix in one process")
parser.add_argument("--subj", type=int, nargs="+", default=[1], choices=[1, 2, 5, 7])
parser.add_argument("--conditions", type=str, nargs="+", default=ALL_CONDITIONS,
                    choices=["visionsimple", "visioncomplex", "visionconcepts",
                             "imagerysimple", "imagerycomplex", "imageryconcepts"],
                    help="vision first on purpose: it is the control")
parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                    help="the 10 seeds of the protocol (Kneeland et al., appendix A.2)")
parser.add_argument("--model_name_tpl", type=str,
                    default="prior_257_final_subj{subj:02d}_bimixco_softclip_byol",
                    help="train_logs subfolder of the high-level model, {subj} is substituted")
parser.add_argument("--autoencoder_name_tpl", type=str,
                    default="models/autoencoder_subj{subj:02d}_4x_locont_no_reconst",
                    help="train_logs subfolder of the low-level model, {subj} is substituted")
parser.add_argument("--imagery_dir", type=str, default="../dataset/preprocessed",
                    help="--out_dir of prepare_nsd_imagery.py")
parser.add_argument("--out_dir", type=str, default="imagery",
                    help="where to write the reconstruction tensors")
parser.add_argument("--img2img_strength", type=float, default=.85)
parser.add_argument("--recons_per_sample", type=int, default=16)
parser.add_argument("--vd_cache_dir", type=str, required=True,
                    help="cached Versatile Diffusion (train_logs/models/versatile-diffusion)")
parser.add_argument("--skip_existing", action="store_true",
                    help="do not recompute a (subj, condition, seed) whose .pt is already on disk")
parser.add_argument("--allow_missing_lowlevel", action="store_true",
                    help="upstream silently falls back to img2img_strength=1 when epoch120.pth is "
                         "missing, which also renames the output; here that is an error unless "
                         "this flag is given")
parser.add_argument("--dry_run", action="store_true",
                    help="print the run list and exit")
args = parser.parse_args()

# An unset $VD_DIR expands to the empty string, which argparse accepts: without this
# check the run would try to re-download Versatile Diffusion (~15 GB) into the cwd and
# then die on the scheduler with a confusing offline-mode error.
if not os.path.isdir(args.vd_cache_dir) or not os.path.exists(
        os.path.join(args.vd_cache_dir, "model_index.json")):
    sys.exit(f"--vd_cache_dir={args.vd_cache_dir!r} is not a Versatile Diffusion folder "
             f"(no model_index.json). Did you forget to export VD_DIR?")

os.makedirs(args.out_dir, exist_ok=True)


def recon_path(subj, condition, seed, img2img_strength):
    return (f"{args.out_dir}/recons_subj0{subj}_{condition}_seed{seed}"
            f"_img2img{img2img_strength}_{args.recons_per_sample}samples.pt")


runs = [(s, c, sd) for s in args.subj for c in args.conditions for sd in args.seeds]
print(f"{len(runs)} runs: subj={args.subj} conditions={args.conditions} seeds={args.seeds}")
if args.dry_run:
    for s, c, sd in runs:
        p = recon_path(s, c, sd, args.img2img_strength)
        print(("skip " if args.skip_existing and os.path.exists(p) else "run  ") + p)
    sys.exit(0)


# # Load VD pipe: once for the whole matrix (subject independent)

print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
try:
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(args.vd_cache_dir).to(device).to(torch.float16)
except:
    print("Downloading Versatile Diffusion to", args.vd_cache_dir)
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(
            "shi-labs/versatile-diffusion",
            cache_dir=args.vd_cache_dir).to(device).to(torch.float16)
vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(args.vd_cache_dir, subfolder="scheduler")
num_inference_steps = 20

# Set weighting of Dual-Guidance
text_image_ratio = .0  # .5 means equally weight text and image, 0 means use only image
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

img_variations = False
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)


# # Per-subject models: reloaded only when the subject changes

def load_subject_models(subj):
    """diffusion prior (last.pth) + low-level autoencoder (epoch120.pth) for one subject."""
    num_voxels = NUM_VOXELS[subj]
    print(f"\n=== subj {subj} | num_voxels {num_voxels} ===")

    out_dim = 257 * 768
    voxel2clip = BrainNetwork(**dict(in_dim=num_voxels, out_dim=out_dim))
    voxel2clip.requires_grad_(False)
    voxel2clip.eval()

    prior_network = VersatileDiffusionPriorNetwork(
            dim=768, depth=6, dim_head=64, heads=12,
            causal=False, learned_query_mode="pos_emb")
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=768,
        condition_on_text_encodings=False,
        timesteps=100,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    )

    model_name = args.model_name_tpl.format(subj=subj)
    ckpt_path = os.path.join(f'../train_logs/{model_name}', 'last.pth')
    print("ckpt_path", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("EPOCH: ", checkpoint['epoch'])
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'], strict=False)
    diffusion_prior.eval().to(device)

    autoencoder_name = args.autoencoder_name_tpl.format(subj=subj)
    ae_ckpt_path = os.path.join(f'../train_logs/{autoencoder_name}', 'epoch120.pth')
    img2img_strength = args.img2img_strength
    if os.path.exists(ae_ckpt_path):
        checkpoint = torch.load(ae_ckpt_path, map_location=device)
        voxel2sd = Voxel2StableDiffusionModel(in_dim=num_voxels)
        voxel2sd.load_state_dict(checkpoint['model_state_dict'], strict=False)
        voxel2sd.eval()
        voxel2sd.to(device)
        print("Loaded low-level model!")
    else:
        # Upstream prints and carries on with img2img_strength=1, which silently
        # produces differently named .pt files. In a 200-run batch nobody reads
        # that line in time; stop instead.
        if not args.allow_missing_lowlevel:
            raise SystemExit(f"missing {ae_ckpt_path}\n"
                             f"pass --allow_missing_lowlevel to fall back to img2img_strength=1")
        print("No valid path for low-level model specified; not using img2img!")
        voxel2sd = None
        img2img_strength = 1

    return [diffusion_prior], voxel2sd, img2img_strength, num_voxels


# # One (subject, condition, seed) run: the loop of Reconstructions_imagery.py, verbatim

def run_one(subj, condition, seed, diffusion_priors, voxel2sd, img2img_strength, num_voxels):
    utils.seed_everything(seed=seed)

    from torch.utils.data import TensorDataset
    x_img = torch.load(f"{args.imagery_dir}/subj0{subj}/x_{condition}.pt").float()   # (6, n_reps, V)
    y_img = torch.load(f"{args.imagery_dir}/gt_{condition}.pt").float()              # (6, 3, 425, 425)
    stim_ids = torch.load(f"{args.imagery_dir}/subj0{subj}/stim_ids_{condition}.pt")

    assert x_img.shape[0] == y_img.shape[0] == len(stim_ids), "x / gt / stim_ids length mismatch"
    assert x_img.shape[2] == num_voxels, \
        f"{x_img.shape[2]} voxels in x_{condition}.pt, but subj0{subj} needs {num_voxels}"

    num_val = len(x_img)
    batch_size = val_batch_size = 1   # utils.reconstruction asserts n_samples_save==1
    val_dl = torch.utils.data.DataLoader(TensorDataset(x_img, y_img, stim_ids),
                                         batch_size=val_batch_size, shuffle=False)

    retrieve = False
    plotting = False
    saving = True
    verbose = False
    imsize = 512
    guidance_scale = 3.5

    ind_include = np.arange(num_val)
    all_brain_recons = None
    all_images = None

    only_lowlevel = False
    if img2img_strength == 1:
        img2img = False
    elif img2img_strength == 0:
        img2img = True
        only_lowlevel = True
    else:
        img2img = True

    for val_i, (voxel, img, coco) in enumerate(tqdm(val_dl, total=len(ind_include),
                                                    desc=f"subj{subj} {condition} seed{seed}")):
        if val_i < np.min(ind_include):
            continue
        voxel = torch.mean(voxel, axis=1).to(device)

        with torch.no_grad():
            if img2img:
                ae_preds = voxel2sd(voxel.float())
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half() / 0.18215).sample / 2 + 0.5
                # upstream shows blurry_recons for val_i==0; dropped here, the figures
                # would pile up in memory over 200 runs and plt.show() is a no-op headless.
            else:
                blurry_recons = None

            if only_lowlevel:
                brain_recons = blurry_recons
            else:
                grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction(
                    img, voxel,
                    clip_extractor, unet, vae, noise_scheduler,
                    voxel2clip_cls=None,
                    diffusion_priors=diffusion_priors,
                    text_token=None,
                    img_lowlevel=blurry_recons,
                    num_inference_steps=num_inference_steps,
                    n_samples_save=batch_size,
                    recons_per_sample=args.recons_per_sample,
                    guidance_scale=guidance_scale,
                    img2img_strength=img2img_strength,
                    timesteps_prior=100,
                    seed=seed,
                    retrieve=retrieve,
                    plotting=plotting,
                    img_variations=img_variations,
                    verbose=verbose,
                )
                brain_recons = brain_recons[:, laion_best_picks.astype(np.int8)]

            if all_brain_recons is None:
                all_brain_recons = brain_recons
                all_images = img
            else:
                all_brain_recons = torch.vstack((all_brain_recons, brain_recons))
                all_images = torch.vstack((all_images, img))

        if val_i >= np.max(ind_include):
            break

    all_brain_recons = all_brain_recons.view(-1, 3, imsize, imsize)

    path = recon_path(subj, condition, seed, img2img_strength)
    gt_path = f'{args.out_dir}/gt_{condition}.pt'
    if saving:
        torch.save(all_brain_recons, path)
        if not os.path.exists(gt_path):
            torch.save(all_images, gt_path)
    return path


# # The matrix

t0 = time.time()
failed, done, skipped = [], 0, 0
current_subj, subject_models = None, None

for subj, condition, seed in runs:
    path = recon_path(subj, condition, seed, args.img2img_strength)
    if args.skip_existing and os.path.exists(path):
        skipped += 1
        continue

    if subj != current_subj:
        if subject_models is not None:
            del subject_models
            torch.cuda.empty_cache()
        subject_models = load_subject_models(subj)
        current_subj = subj

    diffusion_priors, voxel2sd, img2img_strength, num_voxels = subject_models
    t1 = time.time()
    try:
        written = run_one(subj, condition, seed, diffusion_priors, voxel2sd, img2img_strength, num_voxels)
        done += 1
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {written}  ({time.time()-t1:.0f}s, "
              f"{done}/{len(runs)-skipped} done, {(time.time()-t0)/60:.0f} min elapsed)", flush=True)
    except Exception as e:
        # same contract as the "|| echo FAILED" of the shell loop: one bad run does
        # not take the batch down.
        failed.append((subj, condition, seed, repr(e)))
        print(f"FAILED subj{subj} {condition} seed{seed}: {e!r}", flush=True)

print(f"\ndone {done}, skipped {skipped}, failed {len(failed)} "
      f"in {(time.time()-t0)/60:.0f} min")
for f in failed:
    print("FAILED", f)
sys.exit(1 if failed else 0)

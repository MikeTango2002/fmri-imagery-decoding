"""NSD-Imagery reconstruction with the "+Decoded Text" branch of Takagi & Nishimoto.

Inputs:  scores_latent and captions per condition (decoded_imagery/), imagery_stimuli_18.pt
Outputs: decoded/image-text-imagery/<subj>/<cond>/samples/{idx:05}_{sample:03}.png

Copy of diffusion_decoding.py with four changes. (1) argparse takes --condition, --imagery_dir, --out_dir, --n_iter
(5 as upstream); --method accepts only `text` (cvpr needs scores_c, absent here). (2) Scores and captions are read per
condition. (3) The NSD-core stims_ave / tr_idx / sharedix block is gone: imgidx runs 0..5 and indexes the score rows;
the ground truth is imagery_stimuli_18.pt (skipped for imageryconcepts). (4) The output path. The rest is the published
configuration (ddim_steps=50, eta=0, strength=0.8, scale=5.0, seed=42). 10 samples = --n_iter 10 (the first 5 equal --n_iter 5).

Usage (venv: .venv):
    CUDA_VISIBLE_DEVICES=0 python diffusion_decoding_imagery.py --subject subj01 --condition visioncomplex \
      --method text --n_iter 10 --gpu 0 --imagery_dir ../../decoded_imagery --out_dir ../../decoded/image-text-imagery
"""
from PIL import Image
import argparse, os, random, time
import pandas as pd
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import trange
from einops import rearrange
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TAKAGI_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(SCRIPT_DIR, '..', 'utils'))
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

CONDITIONS = ["visionsimple", "visioncomplex",
              "imagerysimple", "imagerycomplex", "imageryconcepts"]

# Slice of imagery_stimuli_18.pt holding the ground truth of each condition.
# The conceptual stimuli have none: they are cue words, evaluated only by human raters.
GT_SLICE = {"simple": (0, 6), "complex": (6, 12), "concepts": None}


def load_model_from_config(config, ckpt, gpu, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda(f"cuda:{gpu}")
    model.eval()
    return model

def load_img_from_arr(img_arr):
    image = Image.fromarray(img_arr).convert("RGB")
    w, h = 512, 512
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imgidx",
        type=int,
        default=0,
        help="first stimulus index inside the condition (0..5)",
    )
    parser.add_argument(
        "--imgidx_end",
        type=int,
        default=None,
        help="process [imgidx, imgidx_end) in one process, loading Stable Diffusion "
             "only once. Default: all the rows of the score file (6)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="skip an index whose last sample ({idx:05}_{n_iter-1:03}.png) is on disk",
    )
    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
        help="gpu"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subject",
        required=True,
        type=str,
        default=None,
        help="subject name: subj01 or subj02  or subj05  or subj07 for full-data subjects ",
    )
    parser.add_argument(
        "--method",
        required=True,
        type=str,
        choices=["text"],
        help="text (the +Decoded Text branch, the only one evaluated on NSD-Imagery)",
    )
    parser.add_argument(
        "--condition",
        required=True,
        type=str,
        choices=CONDITIONS,
        help="NSD-Imagery condition",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=5,
        help="samples per stimulus; 5 is upstream, the benchmark protocol wants 10",
    )
    parser.add_argument(
        "--imagery_dir",
        type=str,
        default=os.path.join(TAKAGI_ROOT, 'decoded_imagery'),
        help="--out_dir of decode_imagery_features.py (scores + captions/)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(TAKAGI_ROOT, 'decoded', 'image-text-imagery'),
    )
    parser.add_argument(
        "--stimuli_path",
        type=str,
        default=os.path.join(TAKAGI_ROOT, 'nsd', 'nsddata_stimuli', 'stimuli',
                             'imagery_stimuli_18.pt'),
        help="the (18, 3, 425, 425) target tensor, used only to save {idx:05}_org.png",
    )

    # Set parameters
    opt = parser.parse_args()
    seed_everything(opt.seed)
    gpu = opt.gpu
    method = opt.method
    subject = opt.subject
    condition = opt.condition
    captdir = os.path.join(opt.imagery_dir, subject, 'captions')

    # Load Stable Diffusion Model
    config = os.path.join(SCRIPT_DIR, 'stable-diffusion/configs/stable-diffusion/v1-inference.yaml')
    ckpt = os.path.join(SCRIPT_DIR, 'stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt')
    config = OmegaConf.load(f"{config}")
    torch.cuda.set_device(gpu)
    model = load_model_from_config(config, f"{ckpt}", gpu)

    n_samples = 1
    ddim_steps = 50
    ddim_eta = 0.0
    strength = 0.8
    scale = 5.0
    n_iter = opt.n_iter
    precision = 'autocast'
    precision_scope = autocast if precision == "autocast" else nullcontext
    batch_size = n_samples

    outdir = os.path.join(opt.out_dir, subject, condition)
    os.makedirs(outdir, exist_ok=True)

    sample_path = os.path.join(outdir, "samples")
    os.makedirs(sample_path, exist_ok=True)
    device = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * ddim_steps)
    print(f"target t_enc is {t_enc} steps")

    # The two inputs of the text branch, both decoded from the imagery betas.
    roi_latent = 'early'
    scores_latent = np.load(os.path.join(
        opt.imagery_dir, subject,
        f'{subject}_{roi_latent}_scores_init_latent_{condition}.npy'))
    captions = pd.read_csv(os.path.join(captdir, f'captions_brain_{condition}.csv'),
                           sep='\t', header=None)
    assert len(captions) == len(scores_latent), \
        f"{len(captions)} captions vs {len(scores_latent)} latents for {condition}"

    # Ground truth, saved next to the samples for inspection only: the model never sees it.
    gt_slice = GT_SLICE[condition.replace("vision", "").replace("imagery", "")]
    stimuli = None
    if gt_slice is not None:
        # The row <-> stimulus alignment is asserted upstream, in prepare_nsd_imagery_takagi.py
        # (check 7): row imgidx of the scores is stimulus gt_slice[0] + imgidx.
        stimuli = torch.load(opt.stimuli_path)[gt_slice[0]:gt_slice[1]]
        assert len(stimuli) == len(scores_latent)

    imgidx_end = opt.imgidx_end if opt.imgidx_end is not None else len(scores_latent)

    # Upstream seeds once per process and then samples exactly one image; every
    # invocation reaches this point with an identical RNG state (model instantiation
    # consumes a deterministic number of draws). Snapshot it here and restore it at
    # the top of each index; a batched run draws the same noise upstream would.
    rng_state = (torch.get_rng_state(), torch.cuda.get_rng_state(device),
                 np.random.get_state(), random.getstate())

    for imgidx in range(opt.imgidx, imgidx_end):
        last_sample = os.path.join(sample_path, f"{imgidx:05}_{n_iter - 1:03}.png")
        if opt.skip_existing and os.path.isfile(last_sample):
            print(f"[{imgidx:05}] already done, skipping", flush=True)
            continue
        t_img = time.time()

        torch.set_rng_state(rng_state[0])
        torch.cuda.set_rng_state(rng_state[1], device)
        np.random.set_state(rng_state[2])
        random.setstate(rng_state[3])

        # Ground truth image (skipped for imageryconcepts, which has none)
        if stimuli is not None:
            gt = (stimuli[imgidx].numpy() * 255.).clip(0, 255).astype(np.uint8)
            Image.fromarray(rearrange(gt, 'c h w -> h w c')).save(
                os.path.join(sample_path, f"{imgidx:05}_org.png"))

        # Load z (Image)
        imgarr = torch.Tensor(scores_latent[imgidx,:].reshape(4,40,40)).unsqueeze(0).to('cuda')

        # Generate image from Z
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    x_samples = model.decode_first_stage(imgarr)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        im = Image.fromarray(x_sample.astype(np.uint8)).resize((512,512))
        im = np.array(im)

        init_image = load_img_from_arr(im).to('cuda')
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

        # Load c (Semantics)
        c = model.get_learned_conditioning(captions.iloc[imgidx][0]).to('cuda')

        # Generate image from Z (image) + C (semantics)
        base_count = 0
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    for n in trange(n_iter, desc="Sampling"):
                        uc = model.get_learned_conditioning(batch_size * [""])

                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(
                            os.path.join(sample_path, f"{imgidx:05}_{base_count:03}.png"))
                        base_count += 1

        print(f"[{imgidx:05}] done in {time.time() - t_img:.1f}s", flush=True)


if __name__ == "__main__":
    main()

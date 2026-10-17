"""The eight image metrics of Kneeland et al. 2025 (Tables 1, 3, 4) for every model. Benchmark B only.

Inputs:  --recon_root (layout per --model, or --model template), --gt imagery_stimuli_18.pt
Outputs: --out, one TSV with a row per subj, mode, stim_type, sample and repetition

A verbatim transcription of the authors' harness (`MIRAGE@main src/final_evaluations_mi_multi.ipynb`), cells
5-6, 8-22 and 27, quoted in the code. Nothing is improved: a deviation would break the comparison with the
published tables. The brain columns (cells 24/26) are in `gnet_brain_correlation.py`. Run it in the MindEye2
environment (the MIRAGE one) with `export OPENBLAS_CORETYPE=SkylakeX`. The `- 1` in the 2AFC (cell 8) is a bug
reproduced on purpose: chance is 40.91%, the ceiling 90.91%; `aggregate_imagery_metrics.py` adds 1/11 back.
Gates: `--model gt --gen_rep 1` gives PixCorr = SSIM = 1 and 2AFC = 0.9091; `--roll_gt` is the chance level.

Usage:
    python final_evaluations_imagery.py --model mindeye_I --recon_root RECON_ROOT --gt imagery_stimuli_18.pt \
        --subj 1 2 5 7 --mode vision imagery --gen_rep 10 --out results/baselines/mindeye_I.tsv
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import scipy.spatial.distance  # noqa: F401  (populates sp.spatial.distance)
import scipy as sp
import torch
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from torchvision.models.feature_extraction import create_feature_extractor
import clip

# ---------------------------------------------------------------- constants ---

# The benchmark scope. The 18 NSD-Imagery stimuli are 0-5 simple, 6-11 complex,
# 12-17 conceptual; cell 6 keeps only [:12], because the conceptual stimuli have no
# ground-truth image to score against. The two conditions of a modality are concatenated in
# this order; `sample` 0-5 is simple and 6-11 is complex, the split that yields
# Tables 3 and 4.
MODE_CONDITIONS = {
    "vision":  ["visionsimple", "visioncomplex"],
    "imagery": ["imagerysimple", "imagerycomplex"],
}
N_STIM = 6      # stimuli per condition
N_TEST = 12     # the pool of every two-way comparison: 1 congruent + 11 distractors

METRICS = ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)",
           "InceptionV3", "CLIP", "EffNet-B", "SwAV"]

TARGET_DIM = 512    # cell 5


# --------------------------------------------------------------- preflight ---

def _blas_sanity():
    """np.corrcoef is silently wrong on this server's Sapphire Rapids CPU without
    OPENBLAS_CORETYPE=SkylakeX. PixCorr and all four two-way metrics go through it, and a
    broken GEMM collapses every 2AFC towards chance, which on Takagi and MindEye2 is
    indistinguishable from the expected result. Checked against a plain reduction, which
    does not dispatch to the broken kernel.
    """
    rng = np.random.default_rng(0)
    a = rng.standard_normal((32, 4096))
    r = np.corrcoef(a)
    z = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    for i, j in ((0, 1), (5, 17), (31, 2)):
        manual = float(np.sum(z[i] * z[j]) / a.shape[1])
        if abs(manual - r[i, j]) > 1e-9:
            sys.exit(
                f"BLAS SANITY FAILED: np.corrcoef[{i},{j}] = {r[i, j]:.8f} but the "
                f"elementwise reduction gives {manual:.8f}.\n"
                "Re-run with `export OPENBLAS_CORETYPE=SkylakeX` (see the docstring of _blas_sanity).")
    if not os.environ.get("OPENBLAS_CORETYPE"):
        print("note: OPENBLAS_CORETYPE is unset but the BLAS check passed "
              "(different numpy build?) -- continuing")


# ----------------------------------------------------------------- adapters ---
# Each adapter returns the N_STIM reconstructions of one (subject, condition, sample) as a
# (6, 3, H, W) tensor in [0,1], in stimulus order. Same roots and same layouts as
# gnet_brain_correlation.py:129-184, which were already validated against the disk --
# do not invent new ones.

_to_tensor = transforms.ToTensor()


def _open_png(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(path)
    with PIL.Image.open(path) as img:
        return _to_tensor(img.convert("RGB"))


def _adapter_mindeye_I(root: Path, subj: int, cond: str, sample: int) -> torch.Tensor:
    """recons_subj0N_{cond}_seed{s}_img2img0.85_16samples.pt -> (6, 3, 512, 512) fp16.

    The dtype is preserved: cell 5 casts to float32 only after the (here unnecessary)
    resize; the numbers stay bit-identical to feeding the .pt straight in.
    """
    pattern = str(root / f"recons_subj0{subj}_{cond}_seed{sample}_*.pt")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} matches for {pattern} (expected exactly 1)")
    t = torch.load(hits[0], map_location="cpu")
    if t.shape[0] != N_STIM:
        raise ValueError(f"{hits[0]}: first axis is {t.shape[0]}, expected {N_STIM}")
    return t


def _adapter_brain_diffuser(root: Path, subj: int, cond: str, sample: int) -> torch.Tensor:
    """subj0N/{cond}_seed{s}/{i}.png, i = 0..5 in stimulus order."""
    d = root / f"subj0{subj}" / f"{cond}_seed{sample}"
    return torch.stack([_open_png(d / f"{i}.png") for i in range(N_STIM)])


def _adapter_takagi(root: Path, subj: int, cond: str, sample: int) -> torch.Tensor:
    """subj0N/{cond}/samples/{i:05}_{s:03}.png, the layout of pool_paths() in
    codes/utils/eval_extract_features_imagery.py:52-57. iCNN's recon_B_export uses the
    same layout (at 224 px, which cell 5 upsamples to 512, exactly as Kneeland's own
    iCNN reconstructions were)."""
    d = root / f"subj0{subj}" / cond / "samples"
    return torch.stack([_open_png(d / f"{i:05}_{sample:03}.png") for i in range(N_STIM)])


def _adapter_template(root: Path, subj: int, cond: str, sample: int,
                      template: str = "") -> torch.Tensor:
    """--path_template with {subj}/{subj02}/{cond}/{sample}/{sample03}/{stim}/{stim05}."""
    out = []
    for i in range(N_STIM):
        p = root / template.format(subj=subj, subj02=f"{subj:02d}", cond=cond,
                                   sample=sample, sample03=f"{sample:03d}",
                                   stim=i, stim05=f"{i:05d}")
        out.append(torch.load(p, map_location="cpu") if p.suffix.lower() == ".pt"
                   else _open_png(p))
    return torch.stack(out)


PER_CONDITION_ADAPTERS = {
    "mindeye_I": _adapter_mindeye_I,
    "brain_diffuser": _adapter_brain_diffuser,
    "takagi": _adapter_takagi,
    "icnn": _adapter_takagi,
    "template": _adapter_template,
}

# MindEye2 (and MIRAGE itself) save one monolithic tensor per (subject, mode) instead of a
# directory tree: `{model}_all_recons_{mode}.pt`, shaped (18, gen_rep, 3, H, W). That is the
# native input of cell 5; it is loaded whole and truncated to [:12].
MONOLITHIC_MODELS = {"mindeye_II"}

# The first validation gate (see the module docstring) is a model too: it builds its
# "reconstructions" out of the ground truth itself.
GT_MODELS = {"gt"}


def load_ground_truth(path: Path) -> torch.Tensor:
    """imagery_stimuli_18.pt -> (18, 3, 425, 425) float32 in [0,1].

    Cell 4 loads it with no normalisation; the file is already in [0,1]; the /255 branch
    exists only to fail loudly rather than silently score a [0,255] tensor.
    """
    s = torch.load(path, map_location="cpu").float()
    assert tuple(s.shape) == (18, 3, 425, 425), \
        f"{path} is {tuple(s.shape)}, expected (18, 3, 425, 425)"
    if float(s.max()) > 1.5:
        print(f"  WARNING: {path.name} max is {float(s.max()):.1f} -> dividing by 255 "
              "(Kneeland's cell 4 does not; check the file)")
        s = s / 255.0
    assert -1e-4 <= float(s.min()) and float(s.max()) <= 1.0 + 1e-4, \
        "ground truth outside [0,1]"
    return s


def roll_in_blocks(x: torch.Tensor) -> torch.Tensor:
    """Roll by one inside each block of 6; `stim_type` stays aligned.

    Rolling across the whole 12 would move a simple stimulus into the complex block and make
    the per-type chance tables (Tables 3-4) meaningless.
    """
    return torch.cat([torch.roll(x[:N_STIM], -1, dims=0),
                      torch.roll(x[N_STIM:], -1, dims=0)])


def build_final_recons(args, subj: int, mode: str, gt_true: torch.Tensor) -> torch.Tensor:
    """(N_TEST, gen_rep, 3, H, W) with the stimulus order 0-5 simple, 6-11 complex.

    `gt_true` is the pristine ground truth: --model gt must not see the rolled copy, or the
    chance control would degenerate back into the identity gate.
    """
    if args.model in GT_MODELS:
        return gt_true.unsqueeze(1).repeat(1, args.gen_rep, 1, 1, 1)

    if args.model in MONOLITHIC_MODELS:
        pt = Path(str(args.recon_root)) / args.recons_pt.format(
            subj=subj, subj02=f"{subj:02d}", mode=mode)
        t = torch.load(pt, map_location="cpu")
        assert t.ndim == 5 and t.shape[0] == 18, \
            f"{pt} is {tuple(t.shape)}, expected (18, gen_rep, 3, H, W)"
        assert t.shape[1] >= args.gen_rep, \
            f"{pt} has {t.shape[1]} repetitions, --gen_rep is {args.gen_rep}"
        return t[:N_TEST, :args.gen_rep]

    adapter = PER_CONDITION_ADAPTERS[args.model]
    if args.model == "template":
        if not args.path_template:
            sys.exit("--model template requires --path_template")
        adapter = lambda r, s, c, n: _adapter_template(r, s, c, n, args.path_template)

    root = Path(args.recon_root)
    per_sample = []
    for sample in range(args.gen_rep):
        blocks = [adapter(root, subj, cond, sample) for cond in MODE_CONDITIONS[mode]]
        for b, cond in zip(blocks, MODE_CONDITIONS[mode]):
            assert b.shape[0] == N_STIM, f"{cond} sample {sample}: {b.shape[0]} images"
        per_sample.append(torch.cat(blocks))                 # (12, 3, H, W)
    return torch.stack(per_sample, dim=1)                    # (12, gen_rep, 3, H, W)


def resize_recons(final_recons: torch.Tensor, num_test: int, gen_rep: int) -> torch.Tensor:
    """Cell 5, verbatim: including the absence of any clamp to [0,1]."""
    if final_recons.shape[-1] != TARGET_DIM:
        resize_transform = transforms.Resize((TARGET_DIM, TARGET_DIM))
        final_recons_resized = torch.zeros((num_test, gen_rep, 3, TARGET_DIM, TARGET_DIM))
        for sample in range(num_test):
            for rep in range(gen_rep):
                final_recons_resized[sample, rep] = resize_transform(final_recons[sample, rep])
        final_recons = final_recons_resized
    return final_recons.to(torch.float32)


# =============================================================================
# The metrics. Cells 8-22 of final_evaluations_mi_multi.ipynb, transcribed verbatim.
# The only edits are the ones a module needs and a notebook does not: `device` and the
# models come in as arguments instead of globals.
# =============================================================================

@torch.no_grad()
def two_way_identification(all_recons, all_images, model, preprocess, device,
                           feature_layer=None, return_avg=False):
    """Cell 8. The anchor is the RECONSTRUCTION and the candidates are the 12 ground
    truths (`correlations = r[:, i]`), not the other way round.

    The `- 1` on the success count is the off-by-one documented in the module docstring:
    kept because it is what produced Table 1.
    """
    preds = model(torch.stack([preprocess(recon) for recon in all_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    reals_T = reals.T
    preds_T = preds.T
    r = np.corrcoef(reals_T, preds_T, rowvar=False)

    N = len(all_images)
    r = r[:N, N:]                      # (N, N): r[real, pred]

    congruents = np.diag(r)

    success_counts = []
    total_comparisons = N - 1          # 11

    for i in range(N):
        correlations = r[:, i]
        congruent = congruents[i]
        successes = np.sum(correlations < congruent) - 1
        success_rate = successes / total_comparisons
        success_counts.append(success_rate)

    if return_avg:
        return np.mean(success_counts)
    return success_counts


# --- PixCorr, cell 10 ---
preprocess_pixcorr = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
])


def get_pix_corr(all_images, all_recons, return_avg=False):
    all_images_flattened = preprocess_pixcorr(all_images).reshape(len(all_images), -1).cpu()
    all_recons_flattened = preprocess_pixcorr(all_recons).view(len(all_recons), -1).cpu()

    correlations = []
    for i in range(len(all_images)):
        correlations.append(np.corrcoef(all_images_flattened[i], all_recons_flattened[i])[0][1])
    if return_avg:
        return np.mean(correlations)
    return correlations


# --- SSIM, cell 12 ---
preprocess_ssim = transforms.Compose([
    transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
])


def get_ssim(all_images, all_recons, return_avg=False):
    img_gray = rgb2gray(preprocess_ssim(all_images).permute((0, 2, 3, 1)).cpu())
    recon_gray = rgb2gray(preprocess_ssim(all_recons).permute((0, 2, 3, 1)).cpu())

    ssim_score = []
    for im, rec in zip(img_gray, recon_gray):
        ssim_score.append(structural_similarity(rec, im, multichannel=True,
                                                gaussian_weights=True, sigma=1.5,
                                                use_sample_covariance=False, data_range=1.0))
    if return_avg:
        return np.mean(ssim_score)
    return ssim_score


# --- AlexNet, cell 14 ---
preprocess_alexnet = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- InceptionV3, cell 16 ---
preprocess_inception = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- CLIP, cell 18 ---
preprocess_clip = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

# --- EfficientNet-B1, cell 20 ---
# EfficientNet_B1_Weights.DEFAULT is IMAGENET1K_V2, NOT the V1 that `pretrained=True`
# resolves to. Resize is 255, not 224. Both differ from the Brain-Diffuser port; that is
# one of the known sources of the gap between the two evaluation codes.
preprocess_efficientnet = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --- SwAV, cell 22 ---
preprocess_swav = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Nets:
    """The six feature extractors, loaded once for the whole run.

    Note which ones live on the GPU: cells 14/16/18 call `.to(device)`, cells 20/22 do NOT:
    EffNet and SwAV run on the CPU in the original harness, and stay there here. It costs
    nothing (12 images per repetition) and keeps the arithmetic identical.
    """

    def __init__(self, device):
        self.device = device

        alex_weights = AlexNet_Weights.IMAGENET1K_V1
        self.alex = create_feature_extractor(
            alexnet(weights=alex_weights),
            return_nodes=['features.4', 'features.11']).to(device)
        self.alex.eval().requires_grad_(False)

        self.inception = create_feature_extractor(
            inception_v3(weights=Inception_V3_Weights.DEFAULT),
            return_nodes=['avgpool']).to(device)
        self.inception.eval().requires_grad_(False)

        self.clip_model, _ = clip.load("ViT-L/14", device=device)

        self.eff = create_feature_extractor(
            efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT),
            return_nodes=['avgpool'])
        self.eff.eval().requires_grad_(False)

        swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
        self.swav = create_feature_extractor(swav_model, return_nodes=['avgpool'])
        self.swav.eval().requires_grad_(False)

    def get_alexnet(self, all_images, all_recons, return_avg=False):
        alexnet2 = two_way_identification(all_recons.to(self.device).float(), all_images,
                                          self.alex, preprocess_alexnet, self.device,
                                          'features.4', return_avg=return_avg)
        alexnet5 = two_way_identification(all_recons.to(self.device).float(), all_images,
                                          self.alex, preprocess_alexnet, self.device,
                                          'features.11', return_avg=return_avg)
        return alexnet2, alexnet5

    def get_inceptionv3(self, all_images, all_recons, return_avg=False):
        return two_way_identification(all_recons.float(), all_images.float(),
                                      self.inception, preprocess_inception, self.device,
                                      'avgpool', return_avg=return_avg)

    def get_clip(self, all_images, all_recons, return_avg=False):
        return two_way_identification(all_recons, all_images,
                                      self.clip_model.encode_image, preprocess_clip,
                                      self.device, None, return_avg=return_avg)

    @torch.no_grad()
    def get_efficientnet(self, all_images, all_recons, return_avg=False):
        gt = self.eff(preprocess_efficientnet(all_images))['avgpool']
        gt = gt.reshape(len(gt), -1).cpu().numpy()
        fake = self.eff(preprocess_efficientnet(all_recons))['avgpool']
        fake = fake.reshape(len(fake), -1).cpu().numpy()

        effnet = [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
        if return_avg:
            return np.mean(effnet)
        return effnet

    @torch.no_grad()
    def get_swav(self, all_images, all_recons, return_avg=False):
        gt = self.swav(preprocess_swav(all_images))['avgpool']
        gt = gt.reshape(len(gt), -1).cpu().numpy()
        fake = self.swav(preprocess_swav(all_recons))['avgpool']
        fake = fake.reshape(len(fake), -1).cpu().numpy()

        swav = [sp.spatial.distance.correlation(gt[i], fake[i]) for i in range(len(gt))]
        if return_avg:
            return np.mean(swav)
        return swav


# ------------------------------------------------------------------- run ---

def run_one(args, nets, all_images, gt_true, subj, mode, label):
    """Cell 27 for one (subject, modality): one row per (sample, repetition).

    `all_images` is what the metrics score against (rolled under --roll_gt); `gt_true` is the
    pristine ground truth, used only by --model gt.
    """
    final_recons = build_final_recons(args, subj, mode, gt_true)
    num_test, gen_rep = final_recons.shape[0], final_recons.shape[1]
    assert num_test == N_TEST, f"{num_test} stimuli, expected {N_TEST}"

    native = tuple(final_recons.shape[-2:])
    final_recons = resize_recons(final_recons, num_test, gen_rep)
    print(f"  recons {native} -> {tuple(final_recons.shape)}  "
          f"range [{float(final_recons.min()):.3f}, {float(final_recons.max()):.3f}]")
    if float(final_recons.min()) < -1e-3 or float(final_recons.max()) > 1 + 1e-3:
        print("    note: values outside [0,1]. Kneeland does not clamp, and neither do we.")

    rows = []
    for repetition in range(gen_rep):
        rep_recons = final_recons[:, repetition]

        pixcorr = get_pix_corr(all_images, rep_recons)
        ssim = get_ssim(all_images, rep_recons)
        alexnet2, alexnet5 = nets.get_alexnet(all_images, rep_recons)
        inception = nets.get_inceptionv3(all_images, rep_recons)
        clip_2way = nets.get_clip(all_images, rep_recons)
        effnet = nets.get_efficientnet(all_images, rep_recons)
        swav = nets.get_swav(all_images, rep_recons)

        for sample in range(num_test):
            rows.append({
                "model": label, "subj": subj, "mode": mode,
                "stim_type": "simple" if sample < N_STIM else "complex",
                "sample": sample, "repetition": repetition,
                "PixCorr": pixcorr[sample], "SSIM": ssim[sample],
                "AlexNet(2)": alexnet2[sample], "AlexNet(5)": alexnet5[sample],
                "InceptionV3": inception[sample], "CLIP": clip_2way[sample],
                "EffNet-B": effnet[sample], "SwAV": swav[sample],
            })
        print(f"    rep {repetition}: " + "  ".join(
            f"{m}={np.mean([r[m] for r in rows if r['repetition'] == repetition]):.3f}"
            for m in METRICS))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   choices=sorted(set(PER_CONDITION_ADAPTERS) | MONOLITHIC_MODELS | GT_MODELS))
    p.add_argument("--label", default="", help="name the numbers are filed under "
                                               "(default: --model)")
    p.add_argument("--recon_root", default="",
                   help="not needed for --model gt / gt_rolled")
    p.add_argument("--path_template", default="", help="only with --model template")
    p.add_argument("--recons_pt", default="{mode}/subj{subj02}_all_recons_{mode}.pt",
                   help="only for the monolithic models: path under --recon_root, with "
                        "{subj}/{subj02}/{mode} placeholders")
    p.add_argument("--gt", required=True, help="imagery_stimuli_18.pt")
    p.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7])
    p.add_argument("--mode", nargs="+", default=["vision", "imagery"],
                   choices=sorted(MODE_CONDITIONS))
    p.add_argument("--gen_rep", type=int, default=10,
                   help="reconstructions sampled per stimulus (appendix A.2: 10)")
    p.add_argument("--roll_gt", action="store_true",
                   help="chance level: roll the ground truth by one inside each block of 6, "
                        "keeping the model's real reconstructions")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True, help="one TSV for every requested subj x mode")
    args = p.parse_args()

    label = args.label or args.model
    if args.model not in GT_MODELS and not args.recon_root:
        sys.exit(f"--model {args.model} requires --recon_root")

    _blas_sanity()

    gt_true = load_ground_truth(Path(args.gt))[:N_TEST]      # cell 6
    all_images = roll_in_blocks(gt_true) if args.roll_gt else gt_true
    print(f"ground truth {tuple(all_images.shape)} <- {args.gt}"
          + ("   [ROLLED: this run measures the chance level]" if args.roll_gt else ""))
    if args.roll_gt:
        label += "_chance"

    print("loading the six feature extractors...")
    nets = Nets(args.device)

    rows = []
    for subj in args.subj:
        for mode in args.mode:
            print(f"\n=== {label}  subj0{subj}  {mode} ===")
            rows += run_one(args, nets, all_images, gt_true, subj, mode, label)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "subj", "mode", "stim_type", "sample", "repetition"] + METRICS
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if c in METRICS else str(r[c])
                              for c in cols) + "\n")
    print(f"\nwrote {len(rows)} rows to {out}")

    print("\n--- mean over the 12 stimuli and the "
          f"{args.gen_rep} repetitions (the aggregation of cell 28) ---")
    print("| subj | mode | " + " | ".join(METRICS) + " |")
    print("| --- | --- | " + " | ".join(["---"] * len(METRICS)) + " |")
    for subj in args.subj:
        for mode in args.mode:
            sel = [r for r in rows if r["subj"] == subj and r["mode"] == mode]
            vals = " | ".join(f"{np.mean([r[m] for r in sel]):.4f}" for m in METRICS)
            print(f"| subj0{subj} | {mode} | {vals} |")


if __name__ == "__main__":
    main()

"""Brain correlation (Early Vis. / Higher Vis. / Visual Cortex) for every model of the thesis.

Inputs:  reconstructions per model (RECON_ADAPTERS), one canonical beta source (`subj0N/nsd_imagery.pt`,
         (720, V), the same for all models), GNet weights `gnet_multisubject.pt`, `brain_region_masks.hdf5`
Outputs: a TSV per model (`compute`); a table over TSVs (`report`)

The three GNet columns of Kneeland et al. 2025 (Tables 1, 3, 4), from the authors' notebook cells 24/26: per
(stimulus, sample), the Pearson correlation across the voxels of one ROI between the measured betas (averaged
over repetitions) and the betas GNet predicts from the reconstruction. Early = union V1|V2|V3|V4, not the mean
of four correlations. It lives with MindEye2, which owns the encoder: run it in that environment.
Run `compute --gt_ceiling imagery_stimuli_18.pt` once per subject: it tells a broken voxel alignment from a
chance-level model. Usage:
    python gnet_brain_correlation.py compute --model mindeye_I --recon_root RECON_ROOT --betas_root BETAS_ROOT \
        --stim_meta nsdimagery_stimuli.pkl3 --out results/brain_correlation/mindeye_I_brain_corr.tsv
"""

import argparse
import glob
import inspect
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import PIL.Image
import torch
from torchmetrics import PearsonCorrCoef
from torchvision import transforms

from models import GNet8_Encoder

# torch >= 2.6 defaults `torch.load` to weights_only=True, which rejects
# gnet_multisubject.pt: the checkpoint stores numpy scalars (voxel_roi/val_cc) alongside the
# weights, and models.py:711 loads it with no kwargs. These are on-disk artifacts of this work,
# not untrusted downloads; the pre-2.6 behaviour is restored here rather than by editing
# models.py, which is upstream code shared with training and final_evaluations.py.
if "weights_only" in inspect.signature(torch.load).parameters:
    _torch_load = torch.load

    def _torch_load_compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load(*a, **kw)

    torch.load = _torch_load_compat

# ---------------------------------------------------------------- constants ---

# GNet's per-subject output width (models.py:707). It is the nsdgeneral voxel count of the
# 1.8 mm preparation, i.e. the same V as every model's preprocessed betas. Equality is a
# necessary, not sufficient, condition for the voxel alignment.
GNET_SUBJECT_SIZES = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
                      5: 13039, 6: 17907, 7: 12682, 8: 14386}

# Run types of one NSD-Imagery session, and the runs that make up each condition.
# Same tables as the MindEye1 prepare_nsd_imagery.py, which is the transcription of
# `create_whole_region_imagery_*` / `load_nsd_mental_imagery` in MIRAGE's utils.py.
COND_RUNS = {
    "visionsimple":    ["visA"],
    "visioncomplex":   ["visB"],
    "visionconcepts":  ["visC"],
    "imagerysimple":   ["imgA_1", "imgA_2"],
    "imagerycomplex":  ["imgB_1", "imgB_2"],
    "imageryconcepts": ["imgC_1", "imgC_2"],
}
# Position of each condition's 6 stimuli inside imagery_stimuli_18.pt.
COND_STIM_OFFSET = {"simple": 0, "complex": 6, "concepts": 12}
COND_REPS = {c: (8 if c.startswith("vision") else 16) for c in COND_RUNS}

# The benchmark scope: the 12 stimuli with a visual ground truth, both modalities.
# imageryconcepts is opt-in (--conditions) and is NOT part of the reproduction, see report().
DEFAULT_CONDITIONS = ["imagerysimple", "imagerycomplex", "visionsimple", "visioncomplex"]

N_TRIALS = 720
N_STIM = 6          # stimuli per condition
REGIONS = ["EVC", "higher_vis", "nsd_general"]   # Early / Higher / VisCtx


# ----------------------------------------------------------------- adapters ---
# Each adapter returns the N_STIM reconstructions of one (subject, condition, sample) as a
# list of PIL images, in stimulus order (the same order as the betas: stimulus id ascending).
# They all hand PIL images to GNet.predict, which is what final_evaluations.py:578-582 does
# via transforms.ToPILImage(); the tensor-backed models go through that same call here.

def _pil_from_tensor(t: torch.Tensor) -> PIL.Image.Image:
    """Exactly final_evaluations.py:578-582, plus the fp16->fp32 and clamp the .pt need.

    ToPILImage() multiplies a float tensor by 255 and casts to uint8; a value outside
    [0,1] would wrap around. Reconstruction_Metrics.py:96 clamps the recons for every other
    metric; clamping here keeps the two harnesses on the same pixels.
    """
    return transforms.ToPILImage()(t.detach().float().clamp(0, 1))


def _adapter_mindeye_I(root: Path, subj: int, cond: str, sample: int):
    """recons_subj0N_{cond}_seed{s}_img2img0.85_16samples.pt -> (6, 3, 512, 512) fp16."""
    pattern = str(root / f"recons_subj0{subj}_{cond}_seed{sample}_*.pt")
    hits = sorted(glob.glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} matches for {pattern} (expected exactly 1)")
    t = torch.load(hits[0], map_location="cpu")
    if t.shape[0] != N_STIM:
        raise ValueError(f"{hits[0]}: first axis is {t.shape[0]}, expected {N_STIM}")
    return [_pil_from_tensor(t[i]) for i in range(N_STIM)]


def _adapter_brain_diffuser(root: Path, subj: int, cond: str, sample: int):
    """subj0N/{cond}_seed{s}/{i}.png, i = 0..5 in stimulus order."""
    d = root / f"subj0{subj}" / f"{cond}_seed{sample}"
    return [_open_png(d / f"{i}.png") for i in range(N_STIM)]


def _adapter_takagi(root: Path, subj: int, cond: str, sample: int):
    """subj0N/{cond}/samples/{i:05}_{s:03}.png, the layout of pool_paths() in
    codes/utils/eval_extract_features_imagery.py:52-57."""
    d = root / f"subj0{subj}" / cond / "samples"
    return [_open_png(d / f"{i:05}_{sample:03}.png") for i in range(N_STIM)]


def _adapter_mindeye_II(root: Path, subj: int, cond: str, sample: int):
    """finetuned_subj0N_40sess_4096hid/..._all_enhancedrecons_{mode}.pt, monolithic
    (18, gen_rep, 3, 256, 256), already low-level-blended by
    enhanced_recon_inference_imagery.py. `cond` (e.g. "visionsimple") selects the mode
    (vision/imagery) and a 6-stimulus slice (simple=0:6, complex=6:12, concepts=12:18) --
    the same split as recon_inference_imagery.py's x_{mode}all.pt stimulus order."""
    mode = "vision" if cond.startswith("vision") else "imagery"
    offset = {"simple": 0, "complex": 6, "concepts": 12}[cond[len(mode):]]
    m = f"finetuned_subj0{subj}_40sess_4096hid"
    p = root / m / f"{m}_all_enhancedrecons_{mode}.pt"
    t = torch.load(p, map_location="cpu")
    if t.shape[1] <= sample:
        raise IndexError(f"{p}: only {t.shape[1]} reps saved, sample {sample} requested")
    sl = t[offset:offset + N_STIM, sample]
    return [_pil_from_tensor(sl[i]) for i in range(N_STIM)]


def _adapter_template(root: Path, subj: int, cond: str, sample: int, template: str = ""):
    """--path_template with {subj}/{subj02}/{cond}/{sample}/{sample03}/{stim} placeholders.

    Covers a model whose layout is not (yet) hardcoded above, mindeye_II and iCNN, with
    no edit to this file. Example:
        --path_template 'subj{subj02}/{cond}/recon_{stim}_seed{sample}.png'
    """
    imgs = []
    for i in range(N_STIM):
        p = root / template.format(subj=subj, subj02=f"{subj:02d}", cond=cond,
                                   sample=sample, sample03=f"{sample:03d}",
                                   stim=i, stim05=f"{i:05d}")
        imgs.append(_open_png(p) if p.suffix.lower() != ".pt"
                    else _pil_from_tensor(torch.load(p, map_location="cpu")))
    return imgs


def _pre_resize(img: PIL.Image.Image, dim: int) -> PIL.Image.Image:
    """Cell 5 of the authors' notebook, for reconstructions that are not natively 512.

    `final_evaluations_mi_multi.ipynb` brings `final_recons` to 512x512 with
    `transforms.Resize` BEFORE anything else, and only then hands PIL images to GNet, which
    resizes again to 227 with LANCZOS. A model whose reconstructions are already 512 (every
    one of the six models except iCNN) goes through cell 5 as an identity; this flag changes
    nothing for it; iCNN's are 224, and for iCNN the two paths differ:

        Kneeland     224 -> 512 (bilinear, torchvision) -> 227 (LANCZOS)
        without this 224 -> 227 (LANCZOS)

    An upscale followed by a downscale is not the same pixels as one resampling, which is
    the same mechanism already measured on gate 1 of the image harness (max|d| = 0.45,
    mean|d| = 0.0036 per pixel). Whether it matters HERE is a measurement, not a deduction:
    it was measured for iCNN: the means of the brain correlation move by at most 0.006.

    `transforms.Resize` on a PIL image is bilinear by default and antialiases, the same
    call the notebook makes. NOT LANCZOS: cell 5 uses torchvision, not PIL.resize.
    """
    return transforms.Resize((dim, dim))(img)


def _open_png(path: Path) -> PIL.Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    img = PIL.Image.open(path)
    img.load()
    return img


RECON_ADAPTERS = {
    "mindeye_I": _adapter_mindeye_I,
    "brain_diffuser": _adapter_brain_diffuser,
    "takagi": _adapter_takagi,
    "mindeye_II": _adapter_mindeye_II,
    "template": _adapter_template,
}


# ------------------------------------------------------------------- betas ---

def load_trial_metadata(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    exps = np.asarray(d["exps"])
    cues = np.asarray(d["cues"])
    image_map = d["image_map"]
    assert len(exps) == len(cues) == N_TRIALS, \
        f"nsdimagery_stimuli.pkl3 has {len(exps)} trials, expected {N_TRIALS}"
    return exps, cues, image_map


def measured_betas(betas_720: torch.Tensor, cond: str, exps, cues, image_map):
    """(720, V) -> (6, V): select the condition's trials and average over repetitions.

    Same selection as `group_by_stimulus` in the MindEye1 prepare_nsd_imagery.py, then the mean
    over the repetition axis that the harness applies (`average=True`; voxels arrive at
    the correlation already averaged, appendix A.2: "averaged measured brain response").
    """
    idx = np.arange(len(exps))[np.isin(exps, COND_RUNS[cond])]
    stim_of_trial = np.array([image_map[c] for c in cues[idx]])
    x = betas_720[torch.from_numpy(idx).long()]

    stim_ids = np.unique(stim_of_trial)
    assert len(stim_ids) == N_STIM, f"{cond}: {len(stim_ids)} distinct stimuli, expected {N_STIM}"
    out = torch.zeros((N_STIM, x.shape[1]), dtype=torch.float32)
    for k, s in enumerate(stim_ids):
        locs = np.flatnonzero(stim_of_trial == s)
        assert len(locs) == COND_REPS[cond], \
            f"{cond} stimulus {s}: {len(locs)} repetitions, expected {COND_REPS[cond]}"
        out[k] = x[torch.from_numpy(locs).long()].mean(dim=0)
    return out, stim_ids


def load_masks(masks_path: Path, subj: int, n_voxels: int):
    """The three ROI masks, as boolean vectors over the V nsdgeneral voxels."""
    with h5py.File(masks_path, "r") as f:
        g = f[f"subj0{subj}"]
        raw = {k: np.asarray(g[k][:]).astype(bool) for k in
               ("nsd_general", "V1", "V2", "V3", "V4", "higher_vis")}

    for k, m in raw.items():
        assert m.shape == (n_voxels,), \
            (f"mask {k} has {m.shape} entries but the betas have {n_voxels} voxels: "
             "the mask file and the betas are not in the same voxel space")

    evc = raw["V1"] | raw["V2"] | raw["V3"] | raw["V4"]
    masks = {"EVC": evc, "higher_vis": raw["higher_vis"], "nsd_general": raw["nsd_general"]}

    # Sanity of the ROI nesting: early and higher both live inside the visual cortex, and
    # (appendix A.2 of the paper) higher is the set complement of early within it.
    assert (evc & ~raw["nsd_general"]).sum() == 0, "V1|V2|V3|V4 leaks outside nsd_general"
    assert (raw["higher_vis"] & ~raw["nsd_general"]).sum() == 0, "higher_vis leaks outside nsd_general"
    overlap = int((evc & raw["higher_vis"]).sum())
    if overlap:
        print(f"  note: EVC and higher_vis overlap on {overlap} voxels (not disjoint)")
    for k, m in masks.items():
        assert m.sum() > 0, f"mask {k} is empty"
    return masks


# ------------------------------------------------------------------ metric ---

def brain_correlation(betas: torch.Tensor, beta_primes: torch.Tensor, masks: dict):
    """Per-stimulus Pearson across the voxels of each ROI.

    Transcription of final_evaluations.py:591-598: PearsonCorrCoef(num_outputs=n_images)
    applied to (V_roi, n_images) tensors, i.e. one correlation per image computed over the
    voxel axis. Returns {region: (n_images,) tensor}; the averaging is done by the caller,
    so Tables 3-4 are subsets of the same per-stimulus values.
    """
    n = betas.shape[0]
    assert beta_primes.shape == betas.shape, \
        f"beta' is {tuple(beta_primes.shape)} but beta is {tuple(betas.shape)}"
    pec = PearsonCorrCoef(num_outputs=n)
    out = {}
    for region, mask in masks.items():
        m = torch.from_numpy(mask)
        out[region] = pec(betas[:, m].float().moveaxis(0, 1),
                          beta_primes[:, m].float().moveaxis(0, 1)).detach()
    return out


# ----------------------------------------------------------------- compute ---

def compute(args):
    adapter = RECON_ADAPTERS[args.model]
    if args.model == "template":
        if not args.path_template:
            sys.exit("--model template requires --path_template")
        if not args.label:
            sys.exit("--model template requires --label (the name the numbers go under)")
        adapter = lambda r, s, c, n: _adapter_template(r, s, c, n, args.path_template)
    label = args.label or args.model

    exps, cues, image_map = load_trial_metadata(Path(args.stim_meta))
    recon_root = Path(args.recon_root)
    betas_root = Path(args.betas_root)

    gt_stimuli = None
    if args.gt_ceiling:
        gt_stimuli = torch.load(args.gt_ceiling, map_location="cpu").float()
        assert tuple(gt_stimuli.shape) == (18, 3, 425, 425), \
            f"imagery_stimuli_18.pt is {tuple(gt_stimuli.shape)}"
        if float(gt_stimuli.max()) > 1.5:
            gt_stimuli = gt_stimuli / 255.0

    rows = []
    for subj in args.subj:
        print(f"\n=== subj0{subj} ===")

        beta_file = betas_root / f"subj0{subj}" / "nsd_imagery.pt"
        betas_720 = torch.load(beta_file, map_location="cpu").float()
        v = betas_720.shape[1]
        print(f"  betas {tuple(betas_720.shape)}  <- {beta_file}")

        # --- blocking preflight: voxel alignment ---------------------------------
        assert betas_720.shape[0] == N_TRIALS, \
            f"{beta_file}: {betas_720.shape[0]} trials, expected {N_TRIALS}"
        assert v == GNET_SUBJECT_SIZES[subj], (
            f"subj0{subj}: the betas have {v} voxels but GNet predicts "
            f"{GNET_SUBJECT_SIZES[subj]}. These are not the same voxel space -- STOP. "
            "A mismatch here yields correlations around 0, which on Takagi and MindEye2 is "
            "indistinguishable from the expected result.")
        assert not torch.isnan(betas_720).any(), f"{beta_file}: NaN in the betas"
        masks = load_masks(Path(args.masks), subj, v)
        print(f"  masks  EVC {int(masks['EVC'].sum())} | higher {int(masks['higher_vis'].sum())} "
              f"| nsd_general {int(masks['nsd_general'].sum())} of {v}")

        gnet = GNet8_Encoder(device=args.device, subject=subj, model_path=args.gnet)

        # --- the ceiling check ---------------------------------------------------
        if gt_stimuli is not None:
            gt_pred = gnet.predict([transforms.ToPILImage()(gt_stimuli[i]) for i in range(18)])
            assert gt_pred.shape[1] == v, \
                f"GNet returned {gt_pred.shape[1]} voxels, betas have {v}"
            print("  ceiling check (GNet on the true stimuli vs measured betas):")
            for cond in args.conditions:
                b, stim_ids = measured_betas(betas_720, cond, exps, cues, image_map)
                sc = brain_correlation(b, gt_pred[torch.from_numpy(stim_ids).long()], masks)
                vals = "  ".join(f"{r}={float(sc[r].mean()):+.3f}" for r in REGIONS)
                print(f"    {cond:16s} {vals}")
                for r in REGIONS:
                    for k, sid in enumerate(stim_ids):
                        rows.append(dict(model="_ceiling_gt", subj=subj, condition=cond,
                                         stim=k, stim_global=int(sid), sample=-1,
                                         region=r, corr=float(sc[r][k])))
            print("    ^ if these are ~0 on the VISION conditions the voxel alignment is "
                  "broken: no encoder scores at chance on seen images.")

        # --- the models ----------------------------------------------------------
        for cond in args.conditions:
            b, stim_ids = measured_betas(betas_720, cond, exps, cues, image_map)

            # One GNet call per condition: all samples at once, stimulus-major.
            images, index = [], []
            for sample in range(args.n_samples):
                imgs = adapter(recon_root, subj, cond, sample)
                assert len(imgs) == N_STIM, f"adapter returned {len(imgs)} images for {cond}"
                if args.pre_resize:
                    imgs = [_pre_resize(im, args.pre_resize) for im in imgs]
                images += imgs
                index += [(sample, k) for k in range(N_STIM)]
            beta_primes = gnet.predict(images)
            assert beta_primes.shape == (len(images), v), \
                f"GNet returned {tuple(beta_primes.shape)}, expected {(len(images), v)}"

            per_region = {r: np.zeros((args.n_samples, N_STIM)) for r in REGIONS}
            for sample in range(args.n_samples):
                sel = [i for i, (s, _) in enumerate(index) if s == sample]
                sc = brain_correlation(b, beta_primes[sel], masks)
                for r in REGIONS:
                    per_region[r][sample] = sc[r].numpy()
                    for k, sid in enumerate(stim_ids):
                        rows.append(dict(model=label, subj=subj, condition=cond,
                                         stim=k, stim_global=int(sid), sample=sample,
                                         region=r, corr=float(sc[r][k])))
            vals = "  ".join(f"{r}={per_region[r].mean():+.3f}" for r in REGIONS)
            print(f"  {cond:16s} {vals}")

        del gnet
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("model\tsubj\tcondition\tstim\tstim_global\tsample\tregion\tcorr\n")
        for r in rows:
            f.write(f"{r['model']}\t{r['subj']}\t{r['condition']}\t{r['stim']}\t"
                    f"{r['stim_global']}\t{r['sample']}\t{r['region']}\t{r['corr']:.6f}\n")
    print(f"\nwrote {len(rows)} rows to {out}")
    report_from_rows(rows)


# ------------------------------------------------------------------ report ---

# Aggregation order, identical to every other metric of the benchmark:
# per stimulus -> mean over the 10 samples ->
# mean over the stimuli in scope -> per subject -> mean over the 4 subjects.
SCOPES = [
    ("imagery, all 12",  ["imagerysimple", "imagerycomplex"]),
    ("imagery, simple",  ["imagerysimple"]),
    ("imagery, complex", ["imagerycomplex"]),
    ("vision, all 12",   ["visionsimple", "visioncomplex"]),
    ("vision, simple",   ["visionsimple"]),
    ("vision, complex",  ["visioncomplex"]),
    ("imagery, concepts (EXTRA, not in Kneeland)", ["imageryconcepts"]),
]


def report_from_rows(rows):
    models = sorted({r["model"] for r in rows})
    for model in models:
        mrows = [r for r in rows if r["model"] == model]
        print(f"\n### {model}")
        print("| scope | subject | Early | Higher | VisCtx |")
        print("| --- | --- | --- | --- | --- |")
        for label, conds in SCOPES:
            sel = [r for r in mrows if r["condition"] in conds]
            if not sel:
                continue
            subjects = sorted({r["subj"] for r in sel})
            per_subj = {}
            for s in subjects:
                vals = []
                for region in REGIONS:
                    v = [r["corr"] for r in sel if r["subj"] == s and r["region"] == region]
                    vals.append(float(np.mean(v)) if v else float("nan"))
                per_subj[s] = vals
                print(f"| {label} | subj0{s} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |")
            mean = np.mean([per_subj[s] for s in subjects], axis=0)
            print(f"| {label} | **mean {len(subjects)} subj** | "
                  f"**{mean[0]:.3f}** | **{mean[1]:.3f}** | **{mean[2]:.3f}** |")


def report(args):
    rows = []
    for path in args.tsv:
        with open(path) as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                d = dict(zip(header, line.rstrip("\n").split("\t")))
                rows.append(dict(model=d["model"], subj=int(d["subj"]),
                                 condition=d["condition"], stim=int(d["stim"]),
                                 sample=int(d["sample"]), region=d["region"],
                                 corr=float(d["corr"])))
    report_from_rows(rows)


# -------------------------------------------------------------------- main ---

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compute", help="run GNet and write the per-stimulus TSV")
    c.add_argument("--model", required=True, choices=sorted(RECON_ADAPTERS))
    c.add_argument("--path_template", default="", help="only with --model template")
    c.add_argument("--label", default="", help="name the numbers are filed under "
                                               "(default: --model; required with template)")
    c.add_argument("--recon_root", required=True)
    c.add_argument("--betas_root", required=True,
                   help="dir with subj0N/nsd_imagery.pt -- the canonical (720,V) betas, "
                        "the same for every model (see the module docstring)")
    c.add_argument("--stim_meta", required=True, help="nsdimagery_stimuli.pkl3")
    c.add_argument("--gnet", default="mindeyev2_dataset/gnet_multisubject.pt")
    c.add_argument("--masks", default="mindeyev2_dataset/brain_region_masks.hdf5",
                   help="final_evaluations.py:551 opens this relative to the CWD; here it "
                        "is an explicit path")
    c.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7])
    c.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    c.add_argument("--n_samples", type=int, default=10)
    c.add_argument("--gt_ceiling", default="", help="imagery_stimuli_18.pt: run the "
                                                    "voxel-alignment ceiling check")
    c.add_argument("--pre_resize", type=int, default=0, metavar="DIM",
                   help="resize the reconstructions to DIMxDIM before GNet, reproducing "
                        "cell 5 of the authors' notebook (use 512). A no-op for models "
                        "already native at DIM; it only moves iCNN, which is 224 "
                        "(a second run with this resize changed the brain-correlation "
                        "means by at most 0.006)")
    c.add_argument("--device", default="cuda")
    c.add_argument("--out", required=True)
    c.set_defaults(func=compute)

    r = sub.add_parser("report", help="re-aggregate one or more TSVs")
    r.add_argument("tsv", nargs="+")
    r.set_defaults(func=report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

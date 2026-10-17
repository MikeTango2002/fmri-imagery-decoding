#!/usr/bin/env python3
"""Figure 7.1: NSD-Imagery imagery reconstructions of one subject, ground truth plus six models.

Inputs:  reconstruction trees already on disk (listed in ROOT-relative form in the path helpers below)
Outputs: one PNG grid, one stimulus per row; --contact-sheet MODEL writes all ten samples of one model

Every cell is sample 0 unless --trisect-samples / --bd-samples / --me2-samples say otherwise (a single
integer, one per row, or `<stimulus id>=<sample>` pairs). Choosing samples for one model makes the figure a
best-of-ten against a first draw; the script prints the selection to declare in the caption. The
TRISECT row must come from the final model (strength 0.75, mixing 0.4; Section 5.5), as in Table 7.4.

Usage:
    python figures/make_imagery_figure.py --stimuli 0,3,6,7,10,12,13,14 --trisect-samples 0,1,8,7,2,1,5,5
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJ = Path(__file__).resolve().parents[1]
ROOT = PROJ.parent

# The 18 NSD-Imagery stimuli are laid out 6 per group, in id order: 0-5 simple, 6-11 complex,
# 12-17 conceptual, and the index inside a condition directory is simply `glob_id % 6`. Deriving
# the group arithmetically (instead of looking it up in the chosen rows) is what lets ANY subset
# of the 18 be selected for the grid, and it is what the other five models' paths need too,
# because every one of them is addressed by (condition, local index).
N_STIMULI = 18
GROUP_ORDER = ["simple", "complex", "conceptual"]

# Notes on the stimuli inspected so far; the rest are deliberately blank rather than guessed --
# `--contact-sheet trisect` (all 18 rows) is how they get looked at.
STIM_NOTES = {
    0: "horizontal bar",
    3: "descending diagonal",
    6: "surfer -> person mid-action",
    7: "beach -> coastal scene, the clearest two-group split",
    10: "donuts -> plates of food",
    12: "stripes",
    14: "mammal",
    16: "banana",
}

# the 8 default rows, in order; the conceptual ones are ids 12, 14 and 16 (stripes, mammal, banana).
# --stimuli chooses others.
STIMULI_PAPER = [0, 3, 6, 7, 10, 12, 14, 16]


def stim_group(glob_id: int) -> str:
    """Global stimulus id (0..17) -> stimulus group. The only place the layout is encoded."""
    if not 0 <= glob_id < N_STIMULI:
        raise SystemExit(f"stimulus {glob_id} out of range (0..{N_STIMULI - 1})")
    return GROUP_ORDER[glob_id // 6]


def stimuli_rows(ids) -> list:
    """[global ids] -> [(global id, group)], the row table the rest of the module reads."""
    return [(g, stim_group(g)) for g in ids]


# The selected rows. Rebound from --stimuli in __main__; the default is the 8 rows listed above.
STIMULI = stimuli_rows(STIMULI_PAPER)

GROUP_LABEL = {"simple": "Simple", "complex": "Complex", "conceptual": "Conceptual"}
COND = {"simple": "imagerysimple", "complex": "imagerycomplex", "conceptual": "imageryconcepts"}

COLUMNS = ["ground_truth", "trisect", "brain_diffuser", "mindeye_I", "mindeye_II", "icnn", "takagi"]
COL_LABEL = {
    "ground_truth": "Ground truth",
    "trisect": "TRISECT",
    "brain_diffuser": "Brain-Diffuser",
    "mindeye_I": "MindEye1",
    "mindeye_II": "MindEye2",
    "icnn": "iCNN",
    "takagi": "Takagi et al.",
}

# how many samples of each column are on disk; 1 means "sample 0 is the only thing rendered",
# and asking for another one is a hard error, not a silent fallback to 0.
N_SAMPLES = {
    "ground_truth": 1,
    "trisect": 10,
    "brain_diffuser": 10,
    "mindeye_I": 1,
    "mindeye_II": 10,
    "icnn": 1,
    "takagi": 1,
}
SELECTABLE = [m for m, n in N_SAMPLES.items() if n > 1]

# layout, in pixels (17.8 cm text width -> ~195 px/cm, i.e. ~495 dpi in print)
CELL = 470
GROUP_W = 96          # left strip carrying the rotated stimulus-group name
HEADER_H = 118
GAP = 10
GROUP_GAP = 30        # extra space between stimulus groups
GT_GAP = 34           # ground truth is set apart from the six reconstructions
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# the reconstructions that Table 7.4 scores: strength 0.75, mixing 0.4, one diffusion seed per
# (replica, subject, stimulus).
TRISECT_ROOT_DEFAULT = PROJ / "data/imagery_recons_b3_h0_subj01"


def parse_samples(spec: str | None, model: str) -> dict:
    """SPEC -> {global stimulus id: sample index}, one entry per row of STIMULI.

    Accepted forms: `3` (one value for every row), `0,0,2,0,5,0,1,0` (one per row, in the order
    of STIMULI), `6=2,14=5` (explicit pairs; unlisted rows stay at 0). Missing/empty -> all 0.
    """
    ids = [g for g, _ in STIMULI]
    out = {g: 0 for g in ids}
    if not spec:
        return out

    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if any("=" in p for p in parts):
        if not all("=" in p for p in parts):
            raise SystemExit(f"--{model}-samples: mix of `id=sample` pairs and bare values in {spec!r}")
        for p in parts:
            key, val = p.split("=", 1)
            g = int(key)
            if g not in out:
                raise SystemExit(f"--{model}-samples: stimulus {g} is not a row of the figure "
                                 f"(rows: {ids})")
            out[g] = int(val)
    elif len(parts) == 1:
        out = {g: int(parts[0]) for g in ids}
    elif len(parts) == len(ids):
        out = {g: int(v) for g, v in zip(ids, parts)}
    else:
        raise SystemExit(f"--{model}-samples: {len(parts)} values for {len(ids)} rows -- pass 1 "
                         f"value, {len(ids)} values, or `id=sample` pairs")

    n = N_SAMPLES[model]
    for g, s in out.items():
        if not 0 <= s < n:
            raise SystemExit(f"--{model}-samples: sample {s} for stimulus {g}, but only "
                             f"{n} sample(s) of {COL_LABEL[model]} are on disk (0..{n - 1})")
    return out


def cell_image(model: str, glob_id: int, subject: str, trisect_root: Path,
               sample: int = 0) -> Image.Image:
    group = stim_group(glob_id)
    cond = COND[group]
    idx = glob_id % 6

    if sample and N_SAMPLES[model] == 1:
        raise SystemExit(f"{COL_LABEL[model]}: only sample 0 was ever rendered, cannot show "
                         f"sample {sample}")

    if model == "ground_truth":
        p = ROOT / f"figure/brain_diffuser_imagery/ground_truth/{glob_id}.png"
        return Image.open(p).convert("RGB")

    if model == "trisect":
        p = trisect_root / subject / cond / f"sample{sample:03d}" / f"{idx:05d}.png"
        return Image.open(p).convert("RGB")

    if model == "brain_diffuser":
        p = ROOT / f"figure/brain_diffuser_imagery/vd/{subject}/{cond}_seed{sample}/{idx}.png"
        return Image.open(p).convert("RGB")

    if model == "mindeye_I":
        # side-by-side (ground truth | reconstruction), 2 px borders: keep the right half
        p = ROOT / f"figure/mindeye_I_imagery/{subject}/{cond}_seed0/pairs/{idx:04d}.png"
        im = Image.open(p).convert("RGB")
        w, h = im.size
        half = (w - 6) // 2
        return im.crop((4 + half, 2, 4 + half + half, 2 + half))

    if model == "mindeye_II":
        p = (
            ROOT
            / f"figure/mindeye_II_imagery/finetuned_{subject}_40sess_4096hid/imagery/enhanced"
            / f"stim{glob_id:02d}_rep{sample:02d}.png"
        )
        return Image.open(p).convert("RGB")

    if model == "icnn":
        p = ROOT / f"figure/icnn_imagery/picks/{cond}_{idx:05d}_{subject}_s000.png"
        return Image.open(p).convert("RGB")

    if model == "takagi":
        p = ROOT / f"figure/takagi_imagery/{subject}/{cond}/{idx:05d}_000.png"
        return Image.open(p).convert("RGB")

    raise ValueError(model)


def stamp(canvas: Image.Image, x: int, y: int, text: str) -> None:
    """Small white-on-black sample index in the top-left corner of a cell (debug/picking only)."""
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT_BOLD, 36)
    w = d.textlength(text, font=f)
    d.rectangle([x + 6, y + 6, x + 18 + w, y + 54], fill="black")
    d.text((x + 12, y + 8), text, font=f, fill="white")


def column_x(col: int) -> int:
    """Left edge of column `col`; the ground-truth column is set apart from the models."""
    x = GROUP_W
    for c in range(col):
        x += CELL + (GT_GAP if c == 0 else GAP)
    return x


def row_y(row: int) -> int:
    """Top edge of row `row`, accounting for the wider gap between stimulus groups."""
    y = HEADER_H
    for r in range(row):
        y += CELL
        y += GAP if STIMULI[r][1] == STIMULI[r + 1][1] else GROUP_GAP
    return y


def report_samples(samples: dict) -> None:
    """Print the exact provenance of every cell, and warn when it is no longer a first draw."""
    overridden = {m: s for m, s in samples.items() if any(v for v in s.values())}
    if not overridden:
        print("samples: every cell is sample 0 (first draw) -- the caption claim holds as written")
        return
    print("samples: NOT a uniform first draw -- declare this in the caption")
    for m, s in overridden.items():
        picks = " ".join(f"stim{g}={v}" for g, v in s.items())
        print(f"  {COL_LABEL[m]:<16} {picks}")
    print("  WARNING: three columns (MindEye1, iCNN, Takagi) exist only at sample 0, so any "
          "override makes the comparison best-of-N for some models and first-draw for others.")


def build(subject: str, out: Path, trisect_root: Path, samples: dict, annotate: bool) -> None:
    width = column_x(len(COLUMNS) - 1) + CELL
    height = row_y(len(STIMULI) - 1) + CELL

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f_head = ImageFont.truetype(FONT_BOLD, 48)
    f_group = ImageFont.truetype(FONT_BOLD, 46)

    # column headers, centred over their column
    for c, model in enumerate(COLUMNS):
        label = COL_LABEL[model]
        x = column_x(c)
        w = draw.textlength(label, font=f_head)
        draw.text((x + (CELL - w) / 2, 40), label, font=f_head, fill="black")
        draw.line([(x, HEADER_H - 14), (x + CELL, HEADER_H - 14)], fill="black", width=3)

    # rotated group names in the left strip, centred over the rows they span
    for group in GROUP_ORDER:
        rows = [i for i, (_, g) in enumerate(STIMULI) if g == group]
        if not rows:
            continue          # --stimuli may select none of this group
        y0, y1 = row_y(rows[0]), row_y(rows[-1]) + CELL
        label = GROUP_LABEL[group]
        strip = Image.new("RGB", (int(draw.textlength(label, font=f_group)) + 10, 60), "white")
        ImageDraw.Draw(strip).text((0, 0), label, font=f_group, fill="black")
        strip = strip.rotate(90, expand=True)
        canvas.paste(strip, (6, int((y0 + y1 - strip.size[1]) / 2)))
        draw.line([(GROUP_W - 16, y0), (GROUP_W - 16, y1)], fill="black", width=3)

    for r, (glob_id, _) in enumerate(STIMULI):
        y = row_y(r)
        for c, model in enumerate(COLUMNS):
            s = samples.get(model, {}).get(glob_id, 0)
            im = cell_image(model, glob_id, subject, trisect_root, s).resize(
                (CELL, CELL), Image.LANCZOS
            )
            x = column_x(c)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + CELL - 1, y + CELL - 1], outline="#b0b0b0", width=2)
            if annotate and model != "ground_truth":
                stamp(canvas, x, y, str(s))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"{out}  ({width}x{height} px)")
    report_samples(samples)


def contact_sheet(model: str, subject: str, out: Path, trisect_root: Path) -> None:
    """All samples of ONE model, one stimulus per row, the sheet you pick --*-samples from.

    The leftmost column is the ground truth; each row can be judged against what it is meant
    to be rather than only against its own siblings. The arm the samples come from is printed in
    the title: two arms of the same model at the same operating point differ only by the
    diffusion noise, and a sheet that does not name its arm cannot be told apart from the other.
    """
    n = N_SAMPLES[model]
    if n == 1:
        raise SystemExit(f"{COL_LABEL[model]}: a single sample on disk, nothing to choose from")

    cell = 300
    gap = 8
    gt_gap = 26       # the reference column is set apart from the candidates
    head = 120        # title on its own line, sample indices under it
    left = 230        # wide enough for the longest row label
    gt_x = left
    s0_x = gt_x + cell + gt_gap
    width = s0_x + n * (cell + gap)
    height = head + len(STIMULI) * (cell + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT_BOLD, 34)
    f_small = ImageFont.truetype(FONT_REG, 26)

    arm = trisect_root.name if model == "trisect" else "on-disk samples"
    draw.text((10, 12), f"{COL_LABEL[model]} - {subject} - all {n} samples", font=f, fill="black")
    draw.text((10, 56), f"arm: {arm}", font=f_small, fill="#606060")
    draw.text((gt_x + cell // 2 - 60, head - 46), "ground truth", font=f, fill="black")
    for s in range(n):
        draw.text((s0_x + s * (cell + gap) + cell // 2 - 10, head - 46), str(s), font=f,
                  fill="black")
    for r, (glob_id, group) in enumerate(STIMULI):
        y = head + r * (cell + gap)
        draw.text((10, y + cell // 2 - 30), f"{glob_id} {group[:4]}", font=f, fill="black")
        note = STIM_NOTES.get(glob_id, "")
        if note:
            draw.text((10, y + cell // 2 + 8), note[:24], font=f_small, fill="#606060")
        gt = cell_image("ground_truth", glob_id, subject, trisect_root).resize((cell, cell),
                                                                              Image.LANCZOS)
        canvas.paste(gt, (gt_x, y))
        draw.rectangle([gt_x, y, gt_x + cell - 1, y + cell - 1], outline="#303030", width=3)
        for s in range(n):
            im = cell_image(model, glob_id, subject, trisect_root, s).resize((cell, cell),
                                                                            Image.LANCZOS)
            x = s0_x + s * (cell + gap)
            canvas.paste(im, (x, y))
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], outline="#b0b0b0", width=2)

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"{out}  ({width}x{height} px) -- columns are sample indices, rows are stimulus ids")
    ids = ",".join(f"{g}=0" for g, _ in STIMULI)
    print(f"pick with:  --{model}-samples {ids}")
    print("            (edit the number after each `=`; unlisted stimuli stay at sample 0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="subj01")
    ap.add_argument("--out", default=str(PROJ / "data/briefing_figures/paper_fig2_imagery_subj01.png"))
    ap.add_argument("--trisect-root", default=str(TRISECT_ROOT_DEFAULT),
                    help="root of the TRISECT reconstructions: <root>/<subj>/<cond>/sample<NNN>/")
    ap.add_argument("--trisect-samples", default=None,
                    help="sample (seed) per TRISECT cell: `3`, `0,0,2,0,5,0,1,0` (one per row) "
                         "or `6=2,14=5` (pairs, unlisted rows stay at 0)")
    ap.add_argument("--bd-samples", default=None, help="same, for Brain-Diffuser (seed0-9)")
    ap.add_argument("--me2-samples", default=None, help="same, for MindEye2 (rep00-09)")
    ap.add_argument("--annotate", action="store_true",
                    help="write the sample index in each cell (for checking, not for the paper)")
    ap.add_argument("--contact-sheet", choices=SELECTABLE, default=None,
                    help="render every sample of ONE model instead of the figure, to choose from")
    ap.add_argument("--stimuli", default=None,
                    help="which of the 18 stimuli are the rows: `all`, or a comma-separated list "
                         f"of ids 0..{N_STIMULI - 1} (e.g. `0,3,6,7,10,12,14,16`). Default: the "
                         "paper's 8 for the figure, all 18 for --contact-sheet. Ids are sorted, "
                         "so the three stimulus groups always stay contiguous.")
    args = ap.parse_args()

    if args.stimuli is None:
        selected = list(range(N_STIMULI)) if args.contact_sheet else list(STIMULI_PAPER)
    elif args.stimuli.strip().lower() == "all":
        selected = list(range(N_STIMULI))
    else:
        try:
            selected = sorted({int(x) for x in args.stimuli.split(",") if x.strip()})
        except ValueError:
            raise SystemExit(f"--stimuli: expected `all` or ids separated by commas, got "
                             f"{args.stimuli!r}")
        if not selected:
            raise SystemExit("--stimuli: no stimulus selected")
    # module-level rebinding: build/parse_samples/contact_sheet/row_y all read this global
    STIMULI = stimuli_rows(selected)
    print("stimuli: " + " ".join(f"{g}({stim_group(g)[:4]})" for g in selected)
          + f"  [{len(selected)} rows]")

    if args.contact_sheet:
        contact_sheet(args.contact_sheet, args.subject, Path(args.out), Path(args.trisect_root))
    else:
        samples = {
            "trisect": parse_samples(args.trisect_samples, "trisect"),
            "brain_diffuser": parse_samples(args.bd_samples, "brain_diffuser"),
            "mindeye_II": parse_samples(args.me2_samples, "mindeye_II"),
        }
        build(args.subject, Path(args.out), Path(args.trisect_root), samples, args.annotate)

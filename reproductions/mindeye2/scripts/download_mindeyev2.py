"""Download the MindEye2 training data (HuggingFace `pscotti/mindeyev2`) for the shared1000 benchmark.

Outputs: flat under <data_path>, as Train.py, recon_inference.py and final_evaluations.py expect: wds/subj0X/{train,
         test,new_test}/ shards, betas_all_subj0X_fp32_renorm.hdf5, coco_images_224_float16.hdf5, the SD VAE and
         ConvNeXt-XL checkpoints, evals/*.pt. Use the same folder for --data_path and --cache_dir when training.

This is the README snippet as a resumable CLI. Excluded, as there: train_logs/ (--include_pretrained fetches it) and
most of evals/. The full dataset (8 subjects) is over 100 GB. --subj restricts the per-subject files, but
multi-subject pretraining that excludes subject X needs the other seven.

Usage (from src/, venv active):
    python download_mindeyev2.py --data_path=mindeyev2_dataset [--subj 1 2 5 7]
"""

import argparse
import os
import re

from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID = "pscotti/mindeyev2"
BRANCH = "main"

# Mirror the filters of the upstream README
EXCLUDE_DIRS = ["train_logs", "evals"]
EXCLUDE_FILES = ["human_trials_mindeye2.ipynb", "subj01_annots.npy", "shared1000.npy"]
INCLUDE_SPECIFIC_FILES = [
    "evals/all_images.pt",
    "evals/all_captions.pt",
    "evals/all_git_generated_captions.pt",
]

SUBJ_RE = re.compile(r"subj0([1-8])")


def wanted(file_path: str, include_pretrained: bool) -> bool:
    """Selection logic of the upstream README, with an option to keep train_logs/ too."""
    exclude_dirs = [d for d in EXCLUDE_DIRS if not (include_pretrained and d == "train_logs")]
    in_excluded_dir = any(ex_dir in file_path for ex_dir in exclude_dirs)
    is_specific = file_path in INCLUDE_SPECIFIC_FILES
    is_excluded_file = any(ex_file in file_path for ex_file in EXCLUDE_FILES)
    return (not in_excluded_dir or is_specific) and not is_excluded_file


def subject_ok(file_path: str, subjects) -> bool:
    """
    If a subject filter is given, drop files that are specific to a subject NOT in the
    set. Files with no subject marker (coco images, cache weights, evals) are always kept.
    """
    if subjects is None:
        return True
    m = SUBJ_RE.search(file_path)
    if m is None:
        return True  # shared / non-subject-specific file
    return int(m.group(1)) in subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="mindeyev2_dataset",
        help="Local folder to download into (use the same path for --data_path and --cache_dir when training)",
    )
    parser.add_argument(
        "--subj", type=int, nargs="+", choices=[1, 2, 3, 4, 5, 6, 7, 8], default=None,
        help="Restrict per-subject files (wds + betas) to these subjects. Default: all 8. "
             "Do NOT subset if you plan multi-subject pretraining.",
    )
    parser.add_argument(
        "--include_pretrained", action="store_true",
        help="Also download train_logs/ (the authors' pretrained/finetuned ckpts). Large.",
    )
    args = parser.parse_args()

    subjects = set(args.subj) if args.subj is not None else None
    os.makedirs(args.data_path, exist_ok=True)

    print(f"Listing files in {REPO_ID} ({BRANCH})...")
    files = list_repo_files(REPO_ID, repo_type="dataset", revision=BRANCH)

    to_download = [
        f for f in files
        if wanted(f, args.include_pretrained) and subject_ok(f, subjects)
    ]
    print(f"Selected {len(to_download)}/{len(files)} files "
          f"(subjects={'all' if subjects is None else sorted(subjects)}, "
          f"pretrained={'yes' if args.include_pretrained else 'no'}).\n")

    for i, file_path in enumerate(to_download, 1):
        print(f"[{i}/{len(to_download)}] {file_path}")
        hf_hub_download(
            REPO_ID, filename=file_path, repo_type="dataset", revision=BRANCH,
            local_dir=args.data_path,
        )

    print("\nAll done. Point both --data_path and --cache_dir at:", os.path.abspath(args.data_path))


if __name__ == "__main__":
    main()

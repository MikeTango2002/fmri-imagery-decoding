"""Download the NSD webdataset shards for subjects 1, 2, 5, 7 into the layout Train_MindEye.py expects.

Outputs: <data_path>/webdataset_avg_split/{train,val,test}/<split>_subj0X_N.tar and metadata_subj0X.json

Files already in place are skipped; files downloaded flat into <data_path>/ are moved to the right place first.

Usage (from src/):
    python download_nsd.py --data_path=cache [--subj 1 2 5 7]
"""

import argparse
import os
import shutil
import requests
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main"
ALL_SUBJECTS = [1, 2, 5, 7]

# Upper bounds for shard probing: script stops early on first 404
MAX_TRAIN_SHARDS = 30
MAX_VAL_SHARDS   = 5
MAX_TEST_SHARDS  = 5


def hf_url(rel_path: str) -> str:
    return f"{BASE_URL}/{rel_path}"


def file_exists_on_hf(url: str) -> bool:
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        return r.status_code == 200
    except requests.RequestException:
        return False


def download_file(url: str, dest: str) -> bool:
    """Download url to dest (streaming). Returns True on success, False on 404."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code == 404:
                return False
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                unit_divisor=1024, leave=False,
                desc=os.path.basename(dest),
            ) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
        os.rename(tmp, dest)
        return True
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise e


def migrate_flat_files(subj: int, data_path: str) -> None:
    """
    Move files that were downloaded flat into data_path/ to their proper
    subdirectory. E.g.:
      cache/train_subj01_0.tar  →  cache/webdataset_avg_split/train/train_subj01_0.tar
    """
    mapping = [
        ("train", f"train_subj0{subj}_"),
        ("val",   f"val_subj0{subj}_"),
        ("test",  f"test_subj0{subj}_"),
    ]
    moved = 0
    for split, prefix in mapping:
        dest_dir = os.path.join(data_path, "webdataset_avg_split", split)
        for fname in os.listdir(data_path):
            if fname.startswith(prefix) and fname.endswith(".tar"):
                src = os.path.join(data_path, fname)
                dst = os.path.join(dest_dir, fname)
                if os.path.exists(dst):
                    print(f"    [migrate] {fname} already in place, removing flat copy")
                    os.remove(src)
                else:
                    os.makedirs(dest_dir, exist_ok=True)
                    print(f"    [migrate] {fname} → {os.path.relpath(dst, data_path)}")
                    shutil.move(src, dst)
                moved += 1
    if moved:
        print(f"    Migrated {moved} file(s) for subj0{subj}.")


def download_split(subj: int, split: str, prefix: str,
                   max_shards: int, data_path: str) -> int:
    """Download shards 0..N until 404. Returns count of shards found."""
    dest_dir = os.path.join(data_path, "webdataset_avg_split", split)
    count = 0
    for i in range(max_shards):
        fname = f"{prefix}{i}.tar"
        rel   = f"webdataset_avg_split/{split}/{fname}"
        url   = hf_url(rel)
        dest  = os.path.join(dest_dir, fname)

        if os.path.exists(dest):
            print(f"    {fname}  [already downloaded, skipping]")
            count += 1
            continue

        ok = download_file(url, dest)
        if not ok:
            break  # no more shards for this subject/split
        print(f"    {fname}  [done]")
        count += 1

    return count


def download_subject(subj: int, data_path: str) -> None:
    print(f"\n{'='*50}")
    print(f"  Subject {subj}")
    print(f"{'='*50}")

    # Move any flat files from a previous partial download
    migrate_flat_files(subj, data_path)

    n_train = download_split(subj, "train", f"train_subj0{subj}_", MAX_TRAIN_SHARDS, data_path)
    n_val   = download_split(subj, "val",   f"val_subj0{subj}_",   MAX_VAL_SHARDS,   data_path)
    n_test  = download_split(subj, "test",  f"test_subj0{subj}_",  MAX_TEST_SHARDS,  data_path)

    # metadata JSON
    meta_rel  = f"webdataset_avg_split/metadata_subj0{subj}.json"
    meta_dest = os.path.join(data_path, meta_rel)
    if os.path.exists(meta_dest):
        print(f"    metadata_subj0{subj}.json  [already downloaded, skipping]")
    else:
        print(f"    Downloading metadata...")
        download_file(hf_url(meta_rel), meta_dest)

    print(f"  Subject {subj}: {n_train} train, {n_val} val, {n_test} test shards.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="cache",
        help="Root folder (webdataset_avg_split/ will be created inside it)",
    )
    parser.add_argument(
        "--subj", type=int, nargs="+", choices=ALL_SUBJECTS, default=ALL_SUBJECTS,
        help="Which subjects to download (default: all four — 1 2 5 7)",
    )
    args = parser.parse_args()

    for subj in args.subj:
        download_subject(subj, args.data_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceForensics++ balanced-subset downloader.

Download a *subset* (e.g., 3k) from:
  - original (real)
  - manipulated sequences: Deepfakes, Face2Face, FaceShifter, FaceSwap, NeuralTextures
  - DeepFakeDetection / actors (uses DFDC file list)

Skips already-downloaded files and preserves the FF++ folder structure.

Examples:
  # 3k real + 3k fake (Deepfakes), compression c23
  python ffpp_subset_downloader.py FaceForensics --dataset original   --compression c23 --type videos --pairs 3000 --server EU --seed 42
  python ffpp_subset_downloader.py FaceForensics --dataset Deepfakes  --compression c23 --type videos --pairs 3000 --server EU --seed 42
"""
import argparse
import os
import sys
import json
import time
import random
import tempfile
import urllib.request
from pathlib import Path
from typing import List

from tqdm import tqdm

# ---------------------------------------------------------------------
# Constants (mirrors the official script)
# ---------------------------------------------------------------------
FILELIST_URL = 'misc/filelist.json'
DEEPFAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
ALL_DATASETS = ['original', 'DeepFakeDetection_original', 'Deepfakes',
                'DeepFakeDetection', 'Face2Face', 'FaceShifter', 'FaceSwap',
                'NeuralTextures']
COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


# ---------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description='FaceForensics++ balanced-subset downloader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument('output_path', type=str, help='Output directory root.')

    p.add_argument('-d', '--dataset', type=str, default='all',
                   choices=list(DATASETS.keys()) + ['all'],
                   help='Which dataset to download.')

    p.add_argument('-c', '--compression', type=str, default='c23',
                   choices=COMPRESSION,
                   help='Compression level (raw, c23, c40).')

    p.add_argument('-t', '--type', type=str, default='videos',
                   choices=TYPE,
                   help='File type: videos, masks, models.')

    p.add_argument('--pairs', type=int, default=None,
                   help='Download a random subset of this many *items* '
                        '(videos/masks). If omitted, downloads ALL.')

    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for selecting subsets.')

    p.add_argument('--server', type=str, default='EU',
                   choices=SERVERS,
                   help='Server to download from (try another if slow).')

    args = p.parse_args()

    # Server URLs
    if args.server == 'EU':
        server_url = 'http://canis.vc.in.tum.de:8100/'
    elif args.server == 'EU2':
        server_url = 'http://kaldir.vc.in.tum.de/faceforensics/'
    elif args.server == 'CA':
        server_url = 'http://falas.cmpt.sfu.ca:8100/'
    else:
        raise ValueError('Invalid server')

    args.tos_url = server_url + 'webpage/FaceForensics_TOS.pdf'
    args.base_url = server_url + 'v3/'
    args.deepfakes_model_url = server_url + 'v3/manipulated_sequences/Deepfakes/models/'
    return args


# ---------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------
def reporthook(count, block_size, total_size):
    # Progress bar hook for urllib
    global _dl_start_time
    if count == 0:
        _dl_start_time = time.time()
        return
    duration = time.time() - _dl_start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * max(1e-6, duration)))
    percent = int(progress_size * 100 / max(1, total_size))
    sys.stdout.write(f"\rProgress: {percent:3d}%, {progress_size / (1024*1024):.1f} MB, {speed} KB/s, {int(duration)} s")
    sys.stdout.flush()


def download_file(url: str, out_file: Path, report_progress: bool = False):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        tqdm.write(f"✔ Skipping (exists): {out_file}")
        return
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(out_file.parent))
    os.close(tmp_fd)
    try:
        if report_progress:
            urllib.request.urlretrieve(url, tmp_name, reporthook=reporthook)
            sys.stdout.write("\n")
        else:
            urllib.request.urlretrieve(url, tmp_name)
        Path(tmp_name).replace(out_file)
        tqdm.write(f"⬇  Downloaded: {out_file.name}")
    except Exception as e:
        Path(tmp_name).unlink(missing_ok=True)
        tqdm.write(f"✖ Failed: {out_file.name} ({e})")


def download_many(filenames: List[str], base_url: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for fn in tqdm(filenames, desc=f"Downloading to {output_dir}"):
        url = base_url + fn
        out_path = output_dir / fn
        download_file(url, out_path, report_progress=False)


# ---------------------------------------------------------------------
# File list builders
# ---------------------------------------------------------------------
def load_json(base_url: str, path: str):
    with urllib.request.urlopen(base_url + path) as resp:
        return json.loads(resp.read().decode('utf-8'))


def build_subset_for_original(base_url: str, pairs: int | None, seed: int) -> List[str]:
    """
    Returns a list of video base names (no extension) for the 'original' dataset.
    The upstream filelist.json is a list of pairs; here we flatten and subset.
    """
    file_pairs = load_json(base_url, FILELIST_URL)  # e.g., [["xyz/000", "abc/001"], ...]
    flat = []
    for a, b in file_pairs:
        flat.append(a)
        flat.append(b)
    if pairs is not None and pairs > 0:
        random.seed(seed)
        random.shuffle(flat)
        flat = flat[:pairs]
    return [f + '.mp4' for f in flat]


def build_subset_for_manipulated(base_url: str, dataset_key: str, pairs: int | None, seed: int) -> List[str]:
    """
    For manipulated datasets (Deepfakes, Face2Face, FaceSwap, FaceShifter, NeuralTextures):
    The official script builds list like: "_".join(pair) AND "_".join(pair[::-1]) per pair.
    We’ll replicate that, then randomly subset.
    """
    file_pairs = load_json(base_url, FILELIST_URL)
    names = []
    for a, b in file_pairs:
        names.append('_'.join([a, b]))
        names.append('_'.join([b, a]))
    if pairs is not None and pairs > 0:
        random.seed(seed)
        random.shuffle(names)
        names = names[:pairs]
    return [n + '.mp4' for n in names]


def build_subset_for_dfdc(base_url: str, dataset_key: str, pairs: int | None, seed: int) -> List[str]:
    """
    DeepFakeDetection / actors use a different JSON with direct filenames.
    """
    filepaths = load_json(base_url, DEEPFAKES_DETECTION_URL)
    if 'actors' in DATASETS[dataset_key]:
        lst = filepaths['actors']
    else:
        lst = filepaths['DeepFakesDetection']
    if pairs is not None and pairs > 0:
        random.seed(seed)
        random.shuffle(lst)
        lst = lst[:pairs]
    return [f + '.mp4' for f in lst]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    # TOS confirm (same as original)
    print('By pressing Enter you confirm you agree to the FaceForensics terms of use:')
    print(args.tos_url)
    input('Press Enter to continue, CTRL-C to abort...')

    out_root = Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS

    for ds in datasets:
        dataset_path = DATASETS[ds]

        # Special zip cases for youtube dumps
        if 'original_youtube_videos' in ds:
            print('Downloading original youtube videos ZIP (huge)…')
            suffix = '' if 'info' not in dataset_path else 'info'
            url = args.base_url + '/' + dataset_path
            out_zip = out_root / f'downloaded_videos{suffix}.zip'
            download_file(url, out_zip, report_progress=True)
            continue

        # Paths
        dataset_videos_url = args.base_url + f"{dataset_path}/{args.compression}/{args.type}/"
        dataset_masks_url  = args.base_url + f"{dataset_path}/masks/videos/"

        # Build list of filenames (without extension for logic, add .mp4 in builders)
        if args.type == 'models':
            if ds != 'Deepfakes':
                print(f"Models are only available for 'Deepfakes'. Skipping '{ds}'.")
                continue
            # Deepfakes models per folder
            print(f"Downloading models for Deepfakes → {dataset_path}")
            # Get list of pair folders
            pair_folders = load_json(args.base_url, FILELIST_URL)
            random.seed(args.seed)
            if args.pairs:  # optional subset of folders
                random.shuffle(pair_folders)
                pair_folders = pair_folders[:args.pairs]
            for folder_pair in tqdm(pair_folders, desc="Deepfakes model folders"):
                folder = '_'.join(folder_pair)
                for model_name in DEEPFAKES_MODEL_NAMES:
                    url = args.deepfakes_model_url + folder + '/' + model_name
                    out_dir = out_root / dataset_path / 'models' / folder
                    out_path = out_dir / model_name
                    download_file(url, out_path, report_progress=False)
            continue

        # Regular videos/masks
        print(f"\nDataset: {ds} | type={args.type} | compression={args.compression} | pairs={args.pairs or 'ALL'}")

        if ds in ['DeepFakeDetection', 'DeepFakeDetection_original']:
            filelist = build_subset_for_dfdc(args.base_url, ds, args.pairs, args.seed)
        elif ds == 'original':
            filelist = build_subset_for_original(args.base_url, args.pairs, args.seed)
        else:
            filelist = build_subset_for_manipulated(args.base_url, ds, args.pairs, args.seed)

        if args.type == 'videos':
            out_dir = out_root / dataset_path / args.compression / 'videos'
            base_url = dataset_videos_url
        elif args.type == 'masks':
            if ds == 'FaceShifter':
                print("Masks not available for FaceShifter. Skipping.")
                continue
            if ds == 'original':
                print("Masks not available for 'original'. Skipping.")
                continue
            out_dir = out_root / dataset_path / 'masks' / 'videos'
            base_url = dataset_masks_url
        else:
            # models handled earlier
            continue

        print(f"Output: {out_dir}")
        # filelist already includes .mp4
        download_many(filelist, base_url, out_dir)

    print("\n✅ Done.")

if __name__ == "__main__":
    main()


# # 3k REAL (original)
# python ffpp_subset_downloader.py FaceForensics --dataset original --compression c23 --type videos --pairs 3000 --server EU --seed 42

# # 3k FAKE (choose one manipulation, e.g. Deepfakes)
# python ffpp_subset_downloader.py FaceForensics --dataset Deepfakes --compression c23 --type videos --pairs 3000 --server EU --seed 42

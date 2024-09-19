import os
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
from functools import partial
import click

from tqdm import tqdm
import glob
from typing import List, Union
import numpy as np
import cv2
from lmdb import Environment
import json

if __name__ == "__main__":
    import sys

    sys.path.append("src")
    from csi_sign_language.utils.lmdb_tool import store_data
else:
    from ...utils.lmdb_tool import store_data


@click.command()
@click.option("--data_root", default="./dataset/PHOENIX-2014-T-release-v3")
@click.option("--output_root", default="./dataset/preprocessed/ph14t_lmdb")
@click.option("--n_threads", default=5, help="number of process to create")
@click.option("--specials", default=["<blank>"], multiple=True)
def main(data_root: str, output_dir: str, n_threads: int, specials: List[str]):
    info = dict(
        name="ph14t_lmdb",
        vocab=[],
    )
    _data_root = Path(data_root)
    _annotation_dir = _data_root / "PHOENIX-2014-T/annotations/manual"
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    info_file_path = _output_dir / "info.json"
    for mode in ["test", "dev", "train"]:
        lmdb_database_name = _output_dir / f"{mode}"
        lmdb_env = Environment(str(lmdb_database_name), map_size=int(1e12))

        annotation_file = _annotation_dir / f"PHOENIX-2014-T.{mode}.corpus.csv"
        annotation = pd.read_csv(annotation_file, sep="|")

        feature_root = _data_root / "PHOENIX-2014-T/features/fullFrame-210x260px" / mode

        task = partial(handle_single_data_by_id, feature_root, lmdb_env=lmdb_env)
        ids = annotation["name"].to_list()

        try:
            with Pool(n_threads) as p:
                results = p.imap_unordered(task, ids)
                for result in tqdm(results, total=len(ids)):
                    pass
        finally:
            lmdb_env.close()

        print(f"Finish processing {mode} data")

    print("generating vocab")
    info["vocab"] = generate_vocab(_data_root, specials)
    with info_file_path.open("w") as f:
        json.dump(info, f)


def generate_vocab(data_root: Path, specials: Union[None, List[str]] = None):
    annotation_dir = data_root / "PHOENIX-2014-T/annotations/manual"
    glosses = set()
    for mode in ["test", "dev", "train"]:
        annotation_file = annotation_dir / f"PHOENIX-2014-T.{mode}.corpus.csv"
        annotation = pd.read_csv(annotation_file, sep="|")

        gloss: str
        for gloss in annotation["orth"]:
            glosses.update(gloss.split())

    glosses = list(glosses)
    if specials is not None:
        glosses = specials + glosses
    return glosses


def handle_single_data_by_id(feature_root: Path, id: str, lmdb_env: Environment):
    video = get_video_by_id(feature_root, id)
    store_data(lmdb_env, id, video)
    return True


def get_video_by_id(feature_root: Path, id: str):
    frames = sorted(
        glob.glob(str(feature_root / f"{id}/*.png")),
        key=lambda x: int(
            os.path.splitext(os.path.basename(x))[0].replace("images", "")
        ),
    )
    frames = [cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB) for frame in frames]
    return np.stack(frames, axis=0)


if __name__ == "__main__":
    videos = get_video_by_id(
        Path(
            "dataset/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"
        ),
        "01April_2010_Thursday_heute-6694",
    )
    print(videos.shape)

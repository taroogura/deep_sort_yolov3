import os
import argparse
import subprocess
from glob import glob
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

from PIL import Image

def make_dirlabel_csv(base_label_dir, csv_path):
    label_df = pd.DataFrame()
    label_dirs = glob(os.path.join(base_label_dir, "*"))
    # label_dirs = [x[0] for x in os.walk(base_label_dir)]
    print(label_dirs)
    for label_dir in tqdm(label_dirs):
        label = os.path.basename(label_dir)
        print("label: ", label)
        track_ids = glob(os.path.join(label_dir, "*"))
        # track_ids = [x[0] for x in os.walk(label_dir)]
        per_label_df = pd.DataFrame({"track_id": list (map(lambda x: os.path.basename(x), track_ids))})
        per_label_df["label"] = label
        label_df = label_df.append(per_label_df)
    label_df.to_csv(csv_path)


def combine_label(base_csv_path, label_df):
    base_df = pd.read_csv(base_csv_path)
    base_df = pd.read_csv(label_df)
    for key, track_df in base_df.groupby(["track_id"]):
        track_df["label"] = base_df[base_df["track_id"] == key].label
        label = os.path.basename(label_dir)
        print("label: ", label)
        track_ids = glob(os.path.join(label_dir, "*"))
        # track_ids = [x[0] for x in os.walk(label_dir)]
        per_label_df = pd.DataFrame({"track_id": list (map(lambda x: os.path.basename(x), track_ids))})
        per_label_df["label"] = label
        label_df = label_df.append(per_label_df)
    label_df.to_csv(csv_path)


if __name__ == '__main__':
    '''
    python make_dirlabel_csv.py --labelsdir labelsdir --csvpath csvpath
    ラベリングされたディレクトリにtrackされた一連の画像の入ったディレクトリが入っている元のディレクトリを指定する
    /labelsdir
        /ogura
            /XXXX-YYYY-0001
            /XXXX-YYYY-0002
        /ohiwa
            /XXXX-ZZZZ-0004
            /XXXX-ZZZZ-0005
    '''
    parser = argparse.ArgumentParser(description='make label:imagesdir list csv')
    parser.add_argument('--labelsdir', help='Path to base labels image file.')
    parser.add_argument('--csvpath', help='Path to save label csv file.')
    args = parser.parse_args()
    make_dirlabel_csv(args.labelsdir, args.csvpath)
    

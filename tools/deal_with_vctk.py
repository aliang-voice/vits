#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :deal_with_v.py
# @Time      :2022/12/16 下午2:09
# @Author    :Aliang

from scipy.io.wavfile import read, write
import librosa
import os
from tqdm import tqdm


def transfer_flac_to_wav(path, new_path):
    x, sr = librosa.load(path, sr=22050)
    write(new_path, 22050, x)


dir_path = "/data/dataset/VCTK-Corpus/wav48"
new_dir_path = "/data/dataset/VCTK-Corpus/downsampled_wavs"
for filepath, dirnames, filenames in os.walk(dir_path):
    print(filepath)
    for filename in tqdm(filenames):
        if "wav" in filename:
            file_path = os.path.join(filepath, filename)
            cluster = filepath[-4:]
            new_cluster_path = new_dir_path + "/" + cluster
            new_filename = filename[:8] + ".wav"
            new_file_path = os.path.join(new_cluster_path, new_filename)
            os.makedirs(new_cluster_path, exist_ok=True)
            transfer_flac_to_wav(file_path, new_file_path)

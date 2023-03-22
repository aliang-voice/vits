#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :deal_with_es.py
# @Time      :2023/1/5 下午5:17
# @Author    :Aliang

import pandas as pd
import os
import sys
sys.path.append("../")
from text.cleaners import spanish_cleaner
from tqdm import tqdm
import librosa
from scipy.io.wavfile import write
import warnings
warnings.filterwarnings('ignore')

es_dir = "/data01/code/dataset/es"
data_name = ["train.tsv", "validated.tsv"]
data_dir = "/data01/code/dataset/es/clips"
new_dir_path = "/data01/code/dataset/es/clips"


def transfer_flac_to_wav(path, new_path):
    x, sr = librosa.load(path, sr=22050)
    write(new_path, 22050, x)


def filter_audio(name):
    if os.path.isfile(os.path.join(data_dir, name)):
        return True
    else:
        return False


def deal_with_filelists():
    for name in data_name:
        data_path = os.path.join(es_dir, name)
        data = pd.read_csv(data_path, sep="\t")
        data = data[['path', 'sentence']]
        data["path"] = data["path"].apply(lambda x: x.replace("mp3", "wav"))
        mask = data.apply(lambda x: filter_audio(x['path']), axis=1)
        data = data[mask]
        data['path'] = data['path'].apply(lambda x: "DUMMY3/" + x)
        data.to_csv(f"../filelists/common_audio_text_es_{name}", sep="|", index=False, header=None)
        data["sentence"] = data["sentence"].apply(lambda x: spanish_cleaner(x))
        data.to_csv(f"../filelists/common_audio_text_es_{name}.cleaned", sep="|", index=False, header=None)


def deal_with_audio():
    for filepath, dirnames, filenames in os.walk(data_dir):
        print(filepath)
        for filename in tqdm(filenames):
            if "mp3" in filename:
                file_path = os.path.join(filepath, filename)
                new_file_path = file_path.replace("mp3", "wav")
                if not os.path.isfile(new_file_path):
                    try:
                        transfer_flac_to_wav(file_path, new_file_path)
                    except:
                        continue


def main():
    deal_with_audio()
    deal_with_filelists()


if __name__ == '__main__':
    main()


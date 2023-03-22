#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/17 上午10:23
# @Author  : Aliang

import pandas as pd
import os
import sys
sys.path.append("../")
from text.cleaners import english_cleaners2
from joblib import Parallel, delayed
import multiprocessing
import librosa
from scipy.io.wavfile import write
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

data_name = ["train.tsv", "validated.tsv"]
data_dir = "DUMMY2"
new_dir_path = "/data01/code/dataset/es/clips"


def transfer_flac_to_wav(path, new_path):
    x, sr = librosa.load(path, sr=22050)
    write(new_path, 22050, x)


def filter_audio(name):
    if os.path.isfile(os.path.join(data_dir, name)):
        return True
    else:
        return False


def parall_cleaner(line):
    path, sentence = line.split("|")
    sentence = english_cleaners2(sentence)
    line = "|".join([path, sentence])
    line = line + "\n"
    return line


def deal_with_filelists():
    num_cores = multiprocessing.cpu_count()
    for path in ['../filelists/wehear_audio_text_val.txt', '../filelists/wehear_audio_text_train.txt']:
        with open(path, 'r', encoding='utf-8') as f:
            data = f.readlines()

        results = Parallel(n_jobs=num_cores, verbose=100)(delayed(parall_cleaner)(line) for line in data)
        with open(path + ".cleaned", "w") as f:
            for line in tqdm(results):
                f.write(line)


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


def split_data():
    data = pd.read_csv("../filelists/metadata.csv", sep="|", names=['path', 'sentence'])
    data = data.sample(frac=1)
    train = data[:int(0.9*len(data))]
    val = data[int(0.9*len(data)):]
    train.to_csv(f"../filelists/wehear_audio_text_train.txt", sep="|", index=False, header=None)
    val.to_csv(f"../filelists/wehear_audio_text_val.txt", sep="|", index=False, header=None)


def main():
    # split_data()
    # deal_with_audio()
    deal_with_filelists()


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :vctk.py
# @Time      :2022/12/28 上午10:26
# @Author    :Aliang

import pandas as pd


data = pd.read_csv("../filelists/vctk_audio_sid_text_train_filelist.txt", header=None, sep="|")

print('done')
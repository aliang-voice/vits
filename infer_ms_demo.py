#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :infer_ms_demo.py
# @Time      :2022/12/28 下午2:56
# @Author    :Aliang


import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import wavfile
import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class InferEngine(object):
    def __init__(self):
        self.load_hps()
        self.load_model()

    def load_hps(self):
        self.hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    def load_model(self):
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).cuda()
        _ = self.net_g.eval()
        _ = utils.load_checkpoint("./model/pretrained_vctk.pth", self.net_g, None)

    def get_text(self, text):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def save_wav(self, wav, path, rate):
        wav *= 32767 / max(0.01, np.max(np.abs(wav))) * 0.6
        wavfile.write(path, rate, wav.astype(np.int16))

    def fit(self, text, n):
        stn_tst = self.get_text(text)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([2]).cuda()
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
                0, 0].data.cpu().float().numpy()

        self.save_wav(audio, f"./vits_out/{n}.wav", self.hps.data.sampling_rate)


if __name__ == '__main__':
    ie = InferEngine()
    with open("data/1963489_6327154_Chapter 57 – A Fever   .txt", 'r') as f:
        txt = f.readlines()
    n = 0
    for sen in tqdm(txt):
        logging.info("sentence: {}".format(sen))
        if sen == '\n':
            continue
        elif sen.startswith('_'):
            continue
        else:
            try:
                ie.fit(sen, n)
                n = n + 1
            except Exception as e:
                logging.info(e)

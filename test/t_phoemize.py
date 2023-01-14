#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :t_phoemize.py
# @Time      :2022/12/26 下午3:03
# @Author    :Aliang

from phonemizer import phonemize


text = "时事史释"
phonemes = phonemize(text, language='cmn', backend='espeak', strip=True, preserve_punctuation=True)
print(phonemes)
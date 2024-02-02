#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""split 3 sets: train-val-test"""

import os
import pandas as pd

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")
DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text", "ipa"])

RANDOM_STATE = 42
for i in range(RANDOM_STATE):  # just a personal fun
	DATA = DATA.sample(frac=1, random_state=i, replace=False, ignore_index=True)

TRAIN_SET_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "ngngngan_audio_text_train_filelist.txt")
VAL_SET_FILE   = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "ngngngan_audio_text_val_filelist.txt")
TEST_SET_FILE  = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "ngngngan_audio_text_test_filelist.txt")

# somewhat arbitrary split
N = len(DATA)  # 6.8k samples (≈ 5h audio)
TRAIN_SIZE = .85  # 85%
VAL_SIZE   = .1
TEST_SIZE  = .05
assert TRAIN_SIZE + VAL_SIZE + TEST_SIZE == 1, "Train, validation, and test sizes must add up to 1."
# TODO: Andrew Ng.: make val & test set came from same distribution but maybe different from train set

idx_train = int(N * TRAIN_SIZE)
idx_val   = int(N * (TRAIN_SIZE + VAL_SIZE))

def save_csv(df: pd.DataFrame, filename: str) -> None:
	df.to_csv(filename,            columns=["audio", "text"], sep=FIELD_SEP, index=False, header=False, encoding="utf-8")
	df.to_csv(filename+".cleaned", columns=["audio",  "ipa"], sep=FIELD_SEP, index=False, header=False, encoding="utf-8")

save_csv(DATA.iloc[:idx_train, :],      TRAIN_SET_FILE)
save_csv(DATA.iloc[idx_train:idx_val, :], VAL_SET_FILE)
save_csv(DATA.iloc[idx_val:, :],         TEST_SET_FILE)

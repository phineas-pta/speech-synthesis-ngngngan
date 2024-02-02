#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""normalize then convert to IPA"""

import os
import re
import pandas as pd
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words  # espeak has num2word built-in but it sucks

from _constants import AUDIO_TEXT_FILE_LIST_PATH, FIELD_SEP

if os.name == "nt":
	from phonemizer.backend.espeak.wrapper import EspeakWrapper
	EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
ESPEAK = EspeakBackend("vi", language_switch="remove-flags")  # auto default to north accent

TRANSCRIPTION_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all.txt")
RAW_DATA = pd.read_csv(TRANSCRIPTION_FILE, sep=FIELD_SEP, names=["audio", "text"])


CHU_SO = re.compile(r"^\d+\.?\d+$")  # something quirky when transcribe with whisper
def special_normalize(text: str) -> str:
	txt = text_normalize(text)
	res = []
	for word in txt.split(" "):
		if word == "%":
			res.append("phần trăm")
		elif CHU_SO.match(word) is not None:
			num = int(word.replace(".", "_"))
			res.append(num2words(num, lang="vi").replace("nghìn", "ngàn"))  # bác Ngạn người Bắc nhưng đọc khác
		elif word not in ".,!?&":
			res.append(word)
	return " ".join(res).strip()

RAW_DATA["text"] = RAW_DATA["text"].map(special_normalize)
RAW_DATA["ipa"] = ESPEAK.phonemize(RAW_DATA["text"], strip=True)  # njobs throw error on windows

RAW_DATA["audio"] = RAW_DATA["audio"].radd("../speech-synthesis-ngngngan/data/06-subs/")

SAVE_FILE = os.path.join(AUDIO_TEXT_FILE_LIST_PATH, "_all_normal_ipa.txt")
RAW_DATA.to_csv(SAVE_FILE, sep=FIELD_SEP, index=False, header=False, encoding="utf-8")

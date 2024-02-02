# -*- coding: utf-8 -*-

"""this script, specific to vietnamese, only exist to replace ███/text/cleaners.py"""

import platform
import re
from underthesea import text_normalize
from phonemizer.backend import EspeakBackend
from num2words import num2words  # espeak has num2word built-in but it sucks

if platform.system().lower() == "windows":
	from phonemizer.backend.espeak.wrapper import EspeakWrapper
	EspeakWrapper.set_library(r"C:\Program Files\eSpeak NG\libespeak-ng.dll")
_ESPEAK = EspeakBackend("vi", language_switch="remove-flags")  # auto default to north accent

_CHU_SO = re.compile(r"^\d+$")


def basic_cleaners_ngngngan(text: str) -> str:
	"""normalize then phonemize"""
	txt = text_normalize(text)
	text_list = []
	for word in txt.split(" "):
		if word == "%":
			text_list.extend(["phần", "trăm"])
		elif word == "&":
			text_list.append("và")
		elif _CHU_SO.match(word) is not None:
			num = num2words(int(word), lang="vi").replace("nghìn", "ngàn")  # bác Ngạn người Bắc nhưng đọc khác
			text_list.extend(num.split(" "))
		elif word not in ".,!?":
			text_list.append(word)
	ipa_list = _ESPEAK.phonemize(text_list, strip=True)  # njobs throw error on windows
	return " ".join(ipa_list).strip()

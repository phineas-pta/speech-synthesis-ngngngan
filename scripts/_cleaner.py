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
_ESPEAK = EspeakBackend("vi", preserve_punctuation=True, language_switch="remove-flags", with_stress=True, tie=True)
# "vi" auto default to north accent

_NAM_20xx = re.compile(r"^20\d{2}$")
_CHU_SO = re.compile(r"^\d+$")


def basic_cleaners_ngngngan(text: str) -> str:
	"""normalize then phonemize"""
	txt = text_normalize(text).replace("-", " ")
	text_list = []
	for word in txt.split():
		if word == "%":
			text_list.extend(["phần", "trăm"])
		elif word == "&":
			text_list.append("và")
		elif _NAM_20xx.match(word) is not None:  # bác Ngạn đọc hơi đặc biệt
			num = "hai ngàn không trăm " + num2words(int(word[-2:]), lang="vi")
			text_list.append(num.split(" "))
		elif _CHU_SO.match(word) is not None:
			num = num2words(int(word), lang="vi").replace("nghìn", "ngàn")  # bác Ngạn người Bắc nhưng đọc khác
			text_list.extend(num.split(" "))
		elif word in ".,;!?":
			text_list[-1] += word
		else:
			text_list.append(word)
	ipa_list = _ESPEAK.phonemize(text_list, strip=True)  # njobs throw error on windows
	return " ".join(ipa_list).strip()

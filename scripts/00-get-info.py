#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""get audio info (speakers count) of audio from authentic youtube channel"""

import json
import re
import unicodedata
from yt_dlp import YoutubeDL

from _constants import DRAFT_FILE, SUMMARY_FILE


# get info from all audio
with YoutubeDL({}) as ydl:
	info = ydl.extract_info("https://www.youtube.com/@nguyenngocnganofficial/videos", download=False)
raw_list_vid = {
	el["id"]: {
		# "url": el["webpage_url"],
		"title": el["title"],
		# "upload_date": el["upload_date"],  # already sorted by upload date
		"description": el["description"],
	}
	for el in ydl.sanitize_info(info)["entries"]
}
with open(DRAFT_FILE, "w", encoding="utf-8") as out_file:
	json.dump(raw_list_vid, out_file, ensure_ascii=False, indent="\t")
# see file draft.json to understand how below regex is written


# count speakers based from video description
TXT = "giọng đọc: nguyễn ngọc ngạn"
count2 = re.compile(rf"\n{TXT}(,| &)[^,&\n]+\n")
count3 = re.compile(rf"\n{TXT}, [^,&\n]+,? &")
def count_speakers(txt: str) -> int:
	txt = unicodedata.normalize("NFC", txt).lower()
	if "giọng đọc:" not in txt or TXT not in txt:
		return 0
	elif f"\n{TXT}\n" in txt:
		return 1
	elif bool(count2.search(txt)):
		return 2
	elif bool(count3.search(txt)):
		return 3
	else:
		print(txt)
		raise ValueError("haha")

for el in raw_list_vid.values():
	el["speakers_count"] = count_speakers(el["description"])


# manual correction
for id in [
	"hoNGMejbF20", "IeDjubTdvXY", "m4N42aD6Twg", "9OvlclzngLY", "ZJiM8YGHDPQ",
	"ELbEiuSHSoE", "-7aAnBs8WQc", "CDlujtWFuJE", "DW3aBHcOdW0", "j8YV2HtDZxA",
	"gp2mK345es0", "AYe1QFFl1fo", "oYdp6FZI884", "1xInGnef1tU", "KpNxXYPYm0I",
	"wcs9JaiyCJA", "ZbKrts2AtYQ", "rZnygcVV3vI",
]:
	raw_list_vid[id]["speakers_count"] = 1
for id in [
	"di_2bXe1HBY", "Ov3o1GIImZM", "DSJ48irndJA", "I7yz-VseOpg", "4CiNPLT12-k",
	"Vx5XZkagNcY", "PGcgxzS1r-I", "RY0TFDo4Mh4", "fZotyTwbsG4", "O02-5flXxfA",
	"_G3SOQCe1Qo", "Zz1jVDtZAW0", "mmi2JyT1yE0", "Qp-657PK3i0", "CdJuF8ITjGw",
	"7Zmp2Z4GZHU", "lPS5zgZs608", "oW9gn-PNjiA", "Mdh4rXn83S8", "eZORvKM8yNY",
	"R9QtL0fKmdE", "8n-PGZc1rN8", "JpFYxZ3pPuA", "VwVGbuuK7RE", "zab-SPU9rXY",
	"zyxVtFhAT1o", "Kb6As5SMVK4", "iPsgbAYREXw", "p8Arw15kRds", "wgCW_gx_jBo",
	"o8YbsqNmKUo", "KnW7dX2oRHk", "FYT9wc19hds", "HJU2w4nP6v4", "qWHSve2ALbU",
	"Rvoy3W6TMNw", "JNQkc-y1veo", "_vbmvXVXZOc", "7xegTZn4LWE", "o4Pfg7AkGmo",
	"iqAK6ojqqkk", "iMu5B0zCqXI",
]:
	raw_list_vid[id]["speakers_count"] = 2
for id in ["01WRW7IV1uQ",]:
	raw_list_vid[id]["speakers_count"] = 3


# final list
list_vid = dict(filter(
	lambda el: el[1]["speakers_count"] > 0,
	raw_list_vid.items()
))

for i in [1, 2, 3]:
	print(f"audio with {i} speakers:", sum(value["speakers_count"] == i for value in list_vid.values()))

with open(SUMMARY_FILE, "w", encoding="utf-8") as out_file:
	json.dump(list_vid, out_file, ensure_ascii=False, indent="\t")

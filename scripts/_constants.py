# -*- coding: utf-8 -*-

import os.path
import json

RAW_DATA_PATH      = os.path.join("data", "01-raw")
VAD_DATA_PATH      = os.path.join("data", "02-vad")
DIARIZED_DATA_PATH = os.path.join("data", "03-diarized")
VOICE_DATA_PATH    = os.path.join("data", "04-voices")
MERGED_DATA_PATH   = os.path.join("data", "05-merged")
SUBS_DATA_PATH     = os.path.join("data", "06-subs")

AUDIO_TEXT_FILE_LIST_PATH = os.path.join("data", "99-audio-text-file-list")
FIELD_SEP = "|"

DRAFT_FILE   = os.path.join("data", "draft.json")
SUMMARY_FILE = os.path.join("data", "data.json")
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
	LIST_VID = json.load(f)

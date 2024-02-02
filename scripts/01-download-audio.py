#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""download audio as wav from pre-configured list"""
# see https://github.com/yt-dlp/yt-dlp/blob/master/yt_dlp/YoutubeDL.py

from yt_dlp import YoutubeDL

from _constants import LIST_VID, RAW_DATA_PATH

list_vid = dict(filter(
	lambda el: el[1]["speakers_count"] == 1,
	LIST_VID.items()
))

YDL_OPTS = {
	"outtmpl": "%(id)s.%(ext)s",
	"format": "bestaudio",
	"paths": {"home": RAW_DATA_PATH},
	"concurrent_fragment_downloads": 8,
	# "cookiesfrombrowser": ("chrome",),
	"overwrites": True,
	"windowsfilenames": True,
	"postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
}

with YoutubeDL(YDL_OPTS) as ydl:
	ydl.download(list(list_vid.keys()))

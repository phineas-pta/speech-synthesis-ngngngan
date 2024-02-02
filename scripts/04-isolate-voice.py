#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""separate voice with demucs"""

import os
from glob import glob

# see https://github.com/facebookresearch/demucs/blob/release_v4/demucs/separate.py
# if newer demucs version then need change
from demucs.pretrained import get_model
from demucs.separate import load_track
from demucs.apply import apply_model
from demucs.audio import save_audio

from _constants import LIST_VID, DIARIZED_DATA_PATH, VOICE_DATA_PATH

MODEL = get_model("htdemucs").cuda()
IJK = MODEL.sources.index("vocals")


def isolate_voice(infile: str, outfile: str) -> None:
	audio = load_track(infile, MODEL.audio_channels, MODEL.samplerate)
	audio_ref = audio.mean(0)
	avg, std = audio_ref.mean(), audio_ref.std()
	audio = (audio - avg) / std  # overwrite object to save memory
	sources = apply_model(
		MODEL, audio[None], device="cuda", split=True, progress=True
	).squeeze()[IJK]  # keep only vocal stem to save memory
	sources = sources * std + avg
	save_audio(wav=sources, path=outfile, samplerate=MODEL.samplerate)
	# torch.cuda.empty_cache()


#################################### main #####################################
for el in LIST_VID:
	infolder = os.path.join(DIARIZED_DATA_PATH, el["id"])
	outfolder = os.path.join(VOICE_DATA_PATH, el["id"])
	os.makedirs(outfolder, exist_ok=True)

	for el in glob("*.wav", root_dir=infolder):
		isolate_voice(infile=os.path.join(infolder, f"{el}.wav"), outfile=os.path.join(outfolder, f"{el}.wav"))


# risk of out-of-memory error with `demucs`
# e.g. memory occupancy:
# 	1800 audio seconds (30 min)
# 	× 4 bytes (float32)
# 	× 2 audio channels
# 	× 44100 Hz (sampling rate)
# 	× 6 stems
# 	× 2 ÷ 1024^3
# ≈ 8 GiB
# ref: https://github.com/facebookresearch/demucs/issues/463#issuecomment-1501137787
# in short: tensor size in GiB = audio length in min × 0.236570835113525390625
